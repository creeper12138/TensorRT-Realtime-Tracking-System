import time

"""
PID Controller & Actuator Interface
-----------------------------------
This module implements a standard discrete PID control algorithm and 
a mock actuator interface for simulation purposes.

In a production environment, the 'Actuator' class would interface 
with physical hardware (e.g., servo motors, robotic arms) via serial/USB.
"""

class PIDController:
    def __init__(self, kp, ki, kd, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.prev_error = 0
        self.integral = 0

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# ==========================================
# Hardware Abstraction Layer (HAL)
# ==========================================

class MockActuator:
    """
    Simulation wrapper for the physical end-effector.
    Replaces the proprietary driver calls (e.g., Logitech/Serial) 
    with console logging for debugging.
    """
    @staticmethod
    def move(dx: int, dy: int):
        # 在真实场景中，这里会调用 DLL 或 串口指令
        # In production: lib.send_pulse(dx, dy)
        if dx != 0 or dy != 0:
            print(f"[Actuator] Servo Adjustment -> Vector({dx}, {dy})")

    @staticmethod
    def click(state: int):
        if state:
            print(f"[Actuator] Trigger Signal SENT")

# 为了兼容 main.py 的调用方式，直接暴露函数接口
_actuator = MockActuator()
move = _actuator.move
click = _actuator.click

# 如果你需要保留 PID 计算状态，可以在这里实例化
# pid_x = PIDController(0.4, 0.0, 0.1)
# pid_y = PIDController(0.4, 0.0, 0.1)
