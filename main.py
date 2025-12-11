# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    C_T = 1.465e-7
    p = np.array([1., 2., 3.])
    rotor_param = {'C_T':C_T,
                   'p':p}

    p1 = rotor_param['p'][0]
    p2 = rotor_param['p'][1]
    p3 = rotor_param['p'][2]

    print('p1 = ', p1)
    print('p2 = ', p2)
    print('p3 = ', p3)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
