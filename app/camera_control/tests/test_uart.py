from app.camera_controll.sources.uartapi import Uart

SERIAL = "/dev/ttyTHS1"

def test_status():
    uart = Uart(SERIAL)

    info = uart.get_info()
    
    assert info

def test_relative_x():
    uart = Uart(SERIAL)

    # send float coordinates
    uart.send_relative(6.34, 0)
    uart.send_relative(-6.34, 0)
    
    # small moves
    uart.send_relative(-3, 0)
    uart.send_relative(3, 0)

    # medium moves
    uart.send_relative(55, 0)
    uart.send_relative(-55, 0)

    # large moves
    uart.send_relative(-360, 0)
    uart.send_relative(360, 0)

def test_relative_y():
    uart = Uart(SERIAL)

    # send float coordinates
    uart.send_relative(0, 6.34)
    uart.send_relative(0, -6.34)
    
    # small moves
    uart.send_relative(0, -3)
    uart.send_relative(0, 3)

    # medium moves
    uart.send_relative(0, 25)
    uart.send_relative(0, -25)

    # large moves
    uart.send_relative(0, -60)
    uart.send_relative(0, 60)