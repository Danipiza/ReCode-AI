import pybullet as pb
import pybullet_data


"""
Connect to PyBullet simulation.

Return:
    robot_id (int) : Robot ID.
"""
def setup_simulation():
    
    # establishes a connection to the PyBullet simulation with a GUI. 
    # you can also use p.DIRECT for a non-GUI mode.
    #   useful for running simulations in headless environments.
    pb.connect(pb.GUI)
    # adds the path to the built-in PyBullet data files
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  
    # sets the gravitational acceleration in the simulation environment. 
    # x, y, z directions. only earths gravity
    pb.setGravity(0, 0, -9.81)

    # load environment and robot
    # plane_id = p.loadURDF("plane.urdf")
    
    # loads a robot model (in this case, R2D2) at the position (0, 0, 0.5). 
    robot_id = pb.loadURDF("r2d2.urdf", [0, 0, 0.5])

    # can be used later to control or query the robot's state in the simulation.
    return robot_id

"""Advance the simulation by one time step"""
def step_simulation(): 
    # updates the physics in the simulation
    pb.stepSimulation()

"""Disconnecting from Simulation"""
def close_simulation(): pb.disconnect()
