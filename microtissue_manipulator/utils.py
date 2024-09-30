import keyboard

class ManualRobotMovement:
    def __init__(self, openapi, block_thread = False):
        self.openapi = openapi
        self.positions = []
        self.step = 1

        # Register the key bindings
        keyboard.on_press_key('up', lambda _: self.move_forward())
        keyboard.on_press_key('down', lambda _: self.move_backward())
        keyboard.on_press_key('left', lambda _: self.move_left())
        keyboard.on_press_key('right', lambda _: self.move_right())
        keyboard.on_press_key('pagedown', lambda _: self.move_z_down())
        keyboard.on_press_key('pageup', lambda _: self.move_z_up())
        keyboard.on_press_key('+', lambda _: self.increase_step())
        keyboard.on_press_key('-', lambda _: self.decrease_step())
        keyboard.on_press_key('s', lambda _: self.save_position())
        # Keep the program running until 'q' is pressed
        if block_thread:
            keyboard.wait('q')
            keyboard.unhook_all()

    def position_safeguard(self, position):
        x, y, z = position
        can_move = False
        x_condition = x >=0 and x <= 380
        y_condition = y >=0 and y <= 350
        z_condition = z >=0.1 and z <= 150

        if x_condition and y_condition and z_condition:
            can_move = True
        return can_move

    def move_z_down(self):
        x,y,z = self.openapi.get_position(verbose = False).values()
        potential_position = (x, y, z-self.step)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('z', -self.step)

    def move_z_up(self):
        x,y,z = self.openapi.get_position(verbose = False).values()
        potential_position = (x, y, z+self.step)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('z', self.step)

    def move_forward(self):
        x,y,z = self.openapi.get_position(verbose = False).values()
        potential_position = (x, y+self.step, z)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('y', self.step)

    def move_backward(self):
        x,y,z = self.openapi.get_position(verbose = False).values()
        potential_position = (x, y-self.step, z)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('y', -self.step)

    def move_left(self):
        x,y,z = self.openapi.get_position(verbose = False).values()
        potential_position = (x-self.step, y, z)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('x', -self.step)

    def move_right(self):
        x,y,z = self.openapi.get_position(verbose = False).values()
        potential_position = (x+self.step, y, z)
        if self.position_safeguard(potential_position):
            self.openapi.move_relative('x', self.step)

    def increase_step(self):
        potential_step = self.step * 10
        if potential_step > 10:
            potential_step = self.step + 10
        if potential_step > 50:
            potential_step = 50
        self.step = potential_step

    def decrease_step(self):
        self.step /= 10
        # print(f'Step: {self.step}')

    def save_position(self):
        position = self.openapi.get_position(verbose = False)
        self.positions.append((position['x'], position['y'], position['z']))
        print(f"Saved position: {position}")