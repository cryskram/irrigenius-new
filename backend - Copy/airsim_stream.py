import airsim
import time

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            _client = airsim.MultirotorClient()
            _client.confirmConnection()
            _client.enableApiControl(True)
        except Exception as e:
            _client = None
            raise RuntimeError(f"Failed to connect to AirSim: {e}")
    return _client


def get_drone_position():
    client = _get_client()
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    return {
        "x": pos.y_val,
        "y": pos.x_val,
        "z": pos.z_val,
    }
