import airsim
import time
import numpy as np
import requests

client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

plants_resp = requests.get("http://localhost:8000/get-saved-plants")
plants = plants_resp.json()["plants"]

response = requests.post(
    "http://localhost:8000/predict/drone-path", json={"plants": plants}
)

path = response.json()["path"]

print("RL Path:", path)


origin = client.getMultirotorState().kinematics_estimated.position
x0 = origin.x_val
y0 = origin.y_val

SCALE = 2.4


def to_airsim_coords(p):
    north = x0 + p["y"]
    east = y0 + p["x"]
    down = -15
    return airsim.Vector3r(north, east, down)


def spray_effect(client, position, duration=0.25):
    obj_name = "spray_" + str(time.time()).replace(".", "")

    puff_pos = airsim.Vector3r(position.x_val, position.y_val, position.z_val + 0.5)

    client.simSpawnObject(
        object_name=obj_name,
        asset_name="Sphere",
        pose=airsim.Pose(puff_pos, airsim.to_quaternion(0, 0, 0)),
        scale=airsim.Vector3r(0.1, 0.1, 0.1),
        physics_enabled=False,
    )

    time.sleep(duration)

    client.simDestroyObject(obj_name)


for i, plant in enumerate(path):
    target = to_airsim_coords(plant)
    print(f"Flying to plant {i}: {target}")
    client.moveToPositionAsync(
        target.x_val, target.y_val, target.z_val, velocity=3
    ).join()
    client.hoverAsync().join()
    time.sleep(0.5)

print("Drug Delivery done!")

client.hoverAsync().join()
time.sleep(2)
client.landAsync().join()

client.armDisarm(False)
client.enableApiControl(False)


# import airsim

# client = airsim.MultirotorClient()
# client.confirmConnection()
# print("Connected")
