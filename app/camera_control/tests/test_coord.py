from app.camera_controll.sources import coord_to_angle, coord_to_steps

from pytest import raises

def test_zero():
    pt = (960, 540)
    imgsz = (1920, 1080)
    angles = (100, 60)

    assert coord_to_angle(pt[0], imgsz[0], angles[0]) == 0
    assert coord_to_angle(pt[1], imgsz[1], angles[1]) == 0

def test_value_error():
    pt = [(1960, 1540), (-25, 0)]
    imgsz = (1920, 1080)
    angles = (100, 60)

    with raises(ValueError):
        coord_to_angle(pt[0], imgsz[0], angles[0])
        coord_to_angle(pt[1], imgsz[1], angles[1])

def test_coordinates():
    pt = (250, 250)
    imgsz = (1000, 1000)
    angles = (90, 90)

    target = (22.5, 22.5)

    assert coord_to_angle(pt[0], imgsz[0], angles[0]) == target[0]
    assert coord_to_angle(pt[1], imgsz[1], angles[1]) == target[1]
