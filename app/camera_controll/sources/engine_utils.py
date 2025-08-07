STEPS_PER_DEGREE = 2_000 / 90

def coord_to_steps(coord:int, dim:int, angle:float, steps:float=STEPS_PER_DEGREE) -> int:
    """Convertation axis coordinate to steps for carriage engine.

    Args:
        coord (int): axis coordinate on frame
        dim (int): width or heught of frame
        angle (float): angle of view of axis
        steps (float): steps per gegree for engine

    Returns:
        int: steps for engine
    """
    return int(steps * coord_to_angle(coord, dim, angle))

def coord_to_angle(coord:int, dim:int, angle:float) -> float:
    """Convertation axis coordinate to steps for carriage engine.

    Args:
        coord (int): axis coordinate on frame
        dim (int): width or heught of frame
        angle (float): angle of view of axis
    
    Returns:
        float: angles for 
    """

    return angle * (0.5 - coord / dim)