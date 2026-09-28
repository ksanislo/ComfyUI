"""Run prompt execution in a child process so that its devices can be released.

A device context cannot be destroyed while the process holding it is alive, so a
server that executes prompts in its own process keeps every device it has touched
initialized until it exits. Executing in a child makes them releasable: the child
holds the devices, and ending it hands them back.

The child speaks to the parent over a pickled connection. It reports through the
same ExecutionServer protocol the in-process executor uses, forwarding each
message to the parent to be sent on to clients.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
from multiprocessing.connection import Connection

#Nothing under comfy is imported at module scope: the child runs this module as
#__main__ before it has the server's argv, and comfy.cli_args parses argv when it
#is imported. Those imports are made where they are needed instead.

#Interrupts reach the child as a signal rather than a message, since it is busy
#executing rather than reading the connection when one arrives.
INTERRUPT_SIGNAL = signal.SIGUSR1


#Parent to child
COMMAND_EXECUTE = "execute"
COMMAND_RESET = "reset"

#Child to parent
MESSAGE_SEND = "send"
MESSAGE_QUEUE_UPDATED = "queue_updated"
MESSAGE_DONE = "done"


def redirect_server_output(server_instance, connection: Connection):
    """Send what this server reports to the parent instead of to its own clients.

    The child builds a real server rather than a stand-in, because nodes reach
    for far more of it than the ExecutionServer protocol describes - the queue,
    the managers, the routes - and a stand-in has to grow an attribute every
    time one of them wants something new. Only the two methods that would talk
    to clients are replaced; the clients are attached to the parent.
    """

    def send_sync(event, data, sid=None):
        connection.send((MESSAGE_SEND, event, data, sid))

    def queue_updated():
        #The queue that matters lives in the parent, so it answers this one.
        connection.send((MESSAGE_QUEUE_UPDATED,))

    server_instance.send_sync = send_sync
    server_instance.queue_updated = queue_updated
    return server_instance


def die_with_parent():
    #An executor that outlives its server is worse than one that never started:
    #it holds the devices with nothing able to reach it. Ask the kernel to end
    #it when the parent goes, which covers the parent being killed outright.
    try:
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        logging.warning("Could not ask to be ended along with the server", exc_info=True)

    #The parent may already have gone between the fork and here.
    if os.getppid() == 1:
        os._exit(0)


def child_main(connection: Connection):
    die_with_parent()

    #Imports are deferred until the parent's argv is in place, so the child
    #parses exactly the arguments the server was started with.
    startup = connection.recv()
    sys.argv = startup["argv"]

    import asyncio

    import folder_paths

    #The parent resolved these from extra_model_paths.yaml and the command line;
    #take them wholesale rather than repeating the resolution and risking a
    #child that looks in different places than the server it serves.
    folder_paths.folder_names_and_paths.update(startup["folder_names_and_paths"])
    for name, setter in (("output_directory", folder_paths.set_output_directory),
                         ("input_directory", folder_paths.set_input_directory),
                         ("user_directory", folder_paths.set_user_directory),
                         ("temp_directory", folder_paths.set_temp_directory)):
        if startup.get(name) is not None:
            setter(startup[name])

    import execution
    import hook_breaker_ac10a0
    import nodes
    import comfy.utils
    import server as server_module

    asyncio_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(asyncio_loop)

    import comfy.model_management
    from app.assets.manager import default_asset_manager

    def handle_interrupt(signum, frame):
        comfy.model_management.interrupt_current_processing(True)

    signal.signal(INTERRUPT_SIGNAL, handle_interrupt)

    #Constructing the server registers it as the singleton the nodes look for.
    #It is never started, so it binds nothing and serves no one.
    asset_manager = default_asset_manager()
    execution_server = redirect_server_output(
        server_module.PromptServer(asyncio_loop, asset_manager), connection)
    execution_server.client_id = startup["client_id"]

    hook_breaker_ac10a0.save_functions()
    asyncio_loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=startup["init_custom_nodes"],
        init_api_nodes=startup["init_api_nodes"],
    ))
    hook_breaker_ac10a0.restore_functions()

    def progress_hook(value, total, preview_image, prompt_id=None, node_id=None):
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()
        execution_server.send_sync(
            "progress",
            {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id},
            execution_server.client_id,
        )

    comfy.utils.set_progress_bar_global_hook(progress_hook)

    executor = execution.PromptExecutor(
        execution_server,
        cache_type=startup["cache_type"],
        cache_args=startup["cache_args"],
        asset_manager=asset_manager,
    )

    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            break

        command = message[0]
        if command == COMMAND_EXECUTE:
            _, prompt, prompt_id, extra_data, outputs_to_execute, client_id = message
            execution_server.client_id = client_id
            execution_server.last_prompt_id = prompt_id
            try:
                executor.execute(prompt, prompt_id, extra_data, outputs_to_execute)
                connection.send((MESSAGE_DONE, executor.history_result, executor.success, executor.status_messages))
            except BaseException as exception:
                logging.error("Prompt execution failed in the executor process", exc_info=exception)
                connection.send((MESSAGE_DONE, {"outputs": {}, "meta": {}}, False, []))
        elif command == COMMAND_RESET:
            executor.reset()
        else:
            break


class SubprocessPromptExecutor:
    """Stands in for PromptExecutor, running the real one in a child process.

    Presents the surface prompt_worker uses - execute(), reset(), and the
    history_result/success/status_messages left behind by the last run - so the
    caller does not need to know where execution happened.
    """

    def __init__(self, server, cache_type=False, cache_args=None, asset_manager=None):
        self.server = server
        self.cache_type = cache_type
        self.cache_args = cache_args
        self.asset_manager = asset_manager
        self.process = None
        self.connection = None
        self.lock = threading.Lock()
        self.reset_state()

        import comfy.model_management
        comfy.model_management.add_interrupt_hook(self.interrupt)
        #Belt as well as braces: the child asks the kernel to end it with us, and
        #we end it ourselves on an orderly exit.
        atexit.register(self.shutdown)

    def interrupt(self, value=True):
        """Pass an interrupt on to the process actually running the prompt."""
        if not value:
            return
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.send_signal(INTERRUPT_SIGNAL)
            except Exception:
                pass

    def reset_state(self):
        self.history_result = {"outputs": {}, "meta": {}}
        self.success = True
        self.status_messages = []

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        if self.is_running():
            return

        parent_socket, child_socket = socket.socketpair()
        child_fd = child_socket.fileno()
        os.set_inheritable(child_fd, True)

        self.process = subprocess.Popen(
            [sys.executable, "-m", "comfy_execution.subprocess_executor", str(child_fd)],
            pass_fds=(child_fd,),
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        child_socket.close()

        import folder_paths
        from comfy.cli_args import args

        self.connection = Connection(parent_socket.detach())
        self.connection.send({
            "argv": list(sys.argv),
            "cache_type": self.cache_type,
            "cache_args": self.cache_args,
            "client_id": self.server.client_id,
            "init_custom_nodes": (not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
            "init_api_nodes": not args.disable_api_nodes,
            "folder_names_and_paths": folder_paths.folder_names_and_paths,
            "output_directory": folder_paths.get_output_directory(),
            "input_directory": folder_paths.get_input_directory(),
            "user_directory": folder_paths.get_user_directory(),
            "temp_directory": folder_paths.get_temp_directory(),
        })
        logging.info("Executor process started (pid {})".format(self.process.pid))

    def shutdown(self, timeout=30.0):
        """End the child, releasing every device it holds."""
        with self.lock:
            if self.process is None:
                return False
            process = self.process
            connection = self.connection
            self.process = None
            self.connection = None

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        logging.info("Executor process ended, devices released")
        return True

    def reset(self):
        self.reset_state()
        if self.is_running():
            try:
                self.connection.send((COMMAND_RESET,))
            except Exception:
                self.shutdown()

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        self.reset_state()
        with self.lock:
            self.start()
            connection = self.connection

        client_id = self.server.client_id
        try:
            connection.send((COMMAND_EXECUTE, prompt, prompt_id, extra_data, execute_outputs, client_id))
        except Exception as exception:
            logging.error("Could not reach the executor process", exc_info=exception)
            self.shutdown()
            self.success = False
            return

        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                #The child died mid-prompt; report it rather than hanging.
                logging.error("The executor process ended during execution")
                self.shutdown()
                self.success = False
                return

            kind = message[0]
            if kind == MESSAGE_SEND:
                _, event, data, sid = message
                self.server.send_sync(event, data, sid)
            elif kind == MESSAGE_QUEUE_UPDATED:
                self.server.queue_updated()
            elif kind == MESSAGE_DONE:
                _, self.history_result, self.success, self.status_messages = message
                return


if __name__ == "__main__":
    child_main(Connection(int(sys.argv[1])))
