import carla
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime

# --- 1. CONFIGURATION ---
TARGET_ROWS_PER_CLASS = 2000
CSV_FILENAME = f"carla_perfect_noisy_dataset.csv"

dataset_buffer = []
class_counts = {
    "Normal_Driving": 0, "Sudden_Acceleration": 0, 
    "Harsh_Brake": 0, "Sharp_Turn": 0, "Sudden_Lane_Change": 0
}

def follow_car(spectator, vehicle):
    transform = vehicle.get_transform()
    location = transform.location + transform.get_forward_vector() * -12 + carla.Location(z=5)
    rotation = transform.rotation
    rotation.pitch = -20 
    spectator.set_transform(carla.Transform(location, rotation))

# 🧠 THE NOISE GENERATOR (Matches your journal paper's engine specs!)
def add_spectral_noise(base_val, time_t):
    # Simulates a Honda Civic engine humming at 5Hz and 12Hz, plus road vibration
    spectral = 0.05 * math.sin(2 * math.pi * 5 * time_t) + 0.02 * math.sin(2 * math.pi * 12 * time_t)
    road_jitter = np.random.normal(0, 0.05)
    return base_val + spectral + road_jitter

# --- 3. MAIN SCRIPT ---
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    spectator = world.get_spectator()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05 # 20Hz Sampling
    world.apply_settings(settings)

    # 🚗 REDUCED SPEEDS SO CARS DON'T CRASH!
    # (Label, Throttle, Brake, Steer, Entry_Speed_m_s)
    stunt_sequence = [
        ("Normal_Driving", 0.3, 0.0, 0.0, 10.0),      # Cruise at 36 km/h
        ("Sudden_Acceleration", 1.0, 0.0, 0.0, 2.0),  # Start slow, punch the gas
        ("Harsh_Brake", 0.0, 1.0, 0.0, 18.0),         # Slam brakes from 65 km/h
        ("Sharp_Turn", 0.3, 0.0, 0.9, 10.0),          # Turn hard at 36 km/h
        ("Sudden_Lane_Change", 0.6, 0.0, -0.6, 14.0)  # S-Curve at 50 km/h
    ]

    print(f"🚗 Starting Noisy Data Generation Pipeline...")
    
    while True:
        if all(count >= TARGET_ROWS_PER_CLASS for count in class_counts.values()):
            print("\n🎉 TARGET REACHED FOR ALL CLASSES!")
            break

        for actor in world.get_actors().filter('vehicle.*'):
            if actor.is_alive: actor.destroy()
        
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(world.get_blueprint_library().filter('model3')[0], spawn_points[50])
        if not vehicle: continue
        
        imu = world.spawn_actor(world.get_blueprint_library().find('sensor.other.imu'), carla.Transform(), attach_to=vehicle)
        state = {"event": "Normal_Driving", "active": True}

        def imu_callback(data):
            if not state["active"]: return
            current_event = state["event"]
            if class_counts[current_event] >= TARGET_ROWS_PER_CLASS: return

            v = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            time_t = data.frame * 0.05
            
            # 🔧 THE FIX: Multiply X and Y by -1 to match real-world smartphones!
            # 🔧 THE FIX: Inject Spectral Noise into the raw IMU data!
            ax = add_spectral_noise(np.clip(data.accelerometer.x, -48, 48) * -1.0, time_t)
            ay = add_spectral_noise(np.clip(data.accelerometer.y, -48, 48) * -1.0, time_t)
            az = add_spectral_noise(np.clip(data.accelerometer.z, -48, 48), time_t)

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "accel_x": ax, "accel_y": ay, "accel_z": az, 
                "gyro_x": data.gyroscope.x, "gyro_y": data.gyroscope.y, "gyro_z": data.gyroscope.z, 
                "speed": speed, "event_type": current_event
            }
            dataset_buffer.append(record)
            class_counts[current_event] += 1

            if data.frame % 50 == 0:
                print(f"📊 [DATA] {current_event:<20} | Rows: {class_counts[current_event]}/{TARGET_ROWS_PER_CLASS} | Speed: {speed:.1f} km/h")

        imu.listen(lambda data: imu_callback(data))

        try:
            for s_name, thr, brk, ste, entry_ms in stunt_sequence:
                if class_counts[s_name] >= TARGET_ROWS_PER_CLASS: continue 

                # 1. The Approach (Getting up to speed SAFELY)
                state['event'] = "Normal_Driving" 
                
                # 🔧 THE FIX: Push the car forward locally, not globally sideways!
                forward = vehicle.get_transform().get_forward_vector()
                vehicle.set_target_velocity(carla.Vector3D(forward.x * entry_ms, forward.y * entry_ms, forward.z * entry_ms))
                
                for _ in range(40): 
                    world.tick(); follow_car(spectator, vehicle); time.sleep(0.01)

                # 2. The Maneuver
                state['event'] = s_name
                print(f"🔴 Triggering Event: {s_name} (Collecting Data...)")
                for _ in range(80): 
                    
                    # 🔧 THE S-CURVE FIX: A Lane Change requires steering left, then right!
                    if s_name == "Sudden_Lane_Change":
                        current_steer = ste if _ < 40 else -ste 
                        vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=current_steer))
                    else:
                        vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=ste))
                    
                    world.tick(); follow_car(spectator, vehicle); time.sleep(0.01)
                
                # 3. The Recovery
                state['event'] = "Normal_Driving"
                vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
                for _ in range(20): 
                    world.tick(); follow_car(spectator, vehicle); time.sleep(0.01)

        finally:
            state["active"] = False
            imu.stop(); imu.destroy(); vehicle.destroy()

    df = pd.DataFrame(dataset_buffer)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"\n💾 Saved perfect noisy dataset to {CSV_FILENAME}!")

if __name__ == '__main__':
    main()