from pymycobot.mycobot import MyCobot
import time

mc = MyCobot('/dev/ttyUSB0',1000000)

time.sleep(1)

print(mc.get_coords())
