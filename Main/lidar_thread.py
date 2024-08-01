import numpy as np
from math import cos, sin, radians
from pyrplidar import PyRPlidar

class LidarThread():

    def __init__(self, port, baudrate):
        self.lidar = PyRPlidar()
        self.port = port
        self.baudrate = baudrate
        self.stop_flag = False
        self.distance_values = []

    def run(self):
        self.lidar.connect(port=self.port, baudrate=self.baudrate, timeout=3)
        self.lidar.set_motor_pwm(660)
        scan_generator = self.lidar.force_scan()

        while not self.stop_flag:
            x_coords, y_coords, angles, distances = [], [], [], []

            for count, scan in enumerate(scan_generator()):
                angle, distance = self.parse_scan(scan)
                if distance > 0:
                    x = distance * sin(radians(angle))
                    y = distance * cos(radians(angle))
                    x_coords.append(x)
                    y_coords.append(y)
                    angles.append(angle)
                    distances.append(distance)

                if count >= 360:
                    # Convert lists to numpy arrays
                    x_coords_array = np.array(x_coords)
                    y_coords_array = np.array(y_coords)
                    angles_array = np.array(angles)
                    distances_array = np.array(distances)

                    # Emit or print the data
                    print("Angles:", angles_array)
                    print("Distances:", distances_array)
                    print("X Coordinates:", x_coords_array)
                    print("Y Coordinates:", y_coords_array)

                    # If you need to emit data, use the following line:
                    # self.distance_values.emit(x_coords_array, y_coords_array, distances_array, angles_array)

                    break

        self.cleanup()

    def parse_scan(self, scan):
        line = str(scan).replace('{', ' ').replace('}', ' ').split(',')
        return float(line[2].split(':')[1]), float(line[3].split(':')[1]) / 10

    def cleanup(self):
        self.lidar.stop()
        self.lidar.set_motor_pwm(0)
        self.lidar.disconnect()

    def stop(self):
        self.stop_flag = True
        self.wait()

if __name__ == '__main__':
    lidar = LidarThread(port='/dev/tty.usbserial-0001', baudrate=256000)
    lidar.run()
