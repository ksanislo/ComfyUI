"""DynamicVRAM setup, for every process that loads models onto a device.

The server performs this at startup, and an executor running in a child process
has to perform it too: without it that process falls back to the legacy
ModelPatcher and none of the VRAM headroom asked for on the command line is
applied, which shows up as an allocation failure rather than as a warning.
"""

import logging
import os

import comfy_aimdo.control
from comfy.cli_args import args, enables_dynamic_vram


def init_control():
    """Bring up comfy-aimdo. Call before the device modules are imported."""

    if enables_dynamic_vram():
        simple_vram_headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
        try:
            comfy_aimdo.control.init(simple_vram_headroom=simple_vram_headroom, nvml_pressure=not args.disable_nvml_pressure)
        except TypeError:
            # comfy-aimdo 0.4.10 protocol.
            try:
                comfy_aimdo.control.init(simple_vram_headroom=simple_vram_headroom)
            except TypeError:
                # comfy-aimdo 0.4.9 protocol.
                comfy_aimdo.control.init()

    if os.name == "nt":
        os.environ['MIMALLOC_PURGE_DELAY'] = '0'


def dynamic_vram_supported():
    import comfy.model_management

    if comfy.model_management.is_nvidia():
        return True
    if comfy.model_management.is_amd():
        if comfy.model_management.rocm_version >= (7, 14):
            return True
    return False


def init_devices(console_log_level):
    """Hand the devices and their headroom to comfy-aimdo and select the patcher."""

    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher

    if not (args.enable_dynamic_vram or (enables_dynamic_vram() and dynamic_vram_supported())):
        return

    if (not args.enable_dynamic_vram) and (comfy.model_management.torch_version_numeric < (2, 8)):
        logging.warning("Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later (2.12+ is recommended). Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")
        return

    try:
        aimdo_initialized = comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
    except TypeError:
        # comfy-aimdo 0.4.9 protocol.
        aimdo_initialized = comfy_aimdo.control.init_devices(d.index for d in comfy.model_management.get_all_torch_devices())

    if aimdo_initialized:
        if console_log_level == 'DEBUG':
            comfy_aimdo.control.set_log_debug()
        elif console_log_level == 'DETAIL':
            try:
                comfy_aimdo.control.set_log_detail()
            except AttributeError:
                comfy_aimdo.control.set_log_info()
        elif console_log_level == 'CRITICAL':
            comfy_aimdo.control.set_log_critical()
        elif console_log_level == 'ERROR':
            comfy_aimdo.control.set_log_error()
        elif console_log_level == 'WARNING':
            comfy_aimdo.control.set_log_warning()
        else: #INFO
            comfy_aimdo.control.set_log_info()

        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
        logging.info("DynamicVRAM support detected and enabled")
    else:
        logging.warning("No working comfy-aimdo install detected. DynamicVRAM support disabled. Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")
