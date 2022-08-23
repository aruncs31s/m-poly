

from smbus2 import SMBus
from mlx90614 import MLX90614
import time

def temp():
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    print("Ambient Temperature :", sensor.get_ambient())
    print("Object Temperature :", sensor.get_object_1())
    #print(type(sensor.get_object_1()))
    return(sensor.get_object_1())
    bus.close()
    
    
    #time.sleep(1)

while True:

    t = temp()
    print(t)
    time.sleep(1)
