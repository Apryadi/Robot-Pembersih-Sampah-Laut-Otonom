import picamera

width = 800
height = 600

camera = picamera.PiCamera()
camera.resolution = (width, height)

filename = 'bus.jpg'

camera.capture(filename)
