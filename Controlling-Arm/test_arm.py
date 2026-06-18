from pymycobot.mycobot import MyCobot
import time

# IMPORTANT: CH340 robot port
mc = MyCobot("/dev/ttyUSB1", 1000000)

print("Connecting to robot...")

time.sleep(2)

print("Moving to start position")
mc.send_angles([0, 0, 0, 0, 0, 0], 50)
time.sleep(2)

print("Test move")
mc.send_angles([20, 20, 20, 0, 0, 0], 50)
time.sleep(2)

print("Back to zero")
mc.send_angles([0, 0, 0, 0, 0, 0], 50)

print("Done")
