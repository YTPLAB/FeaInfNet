
def Monotonicity_Modify_Rate(x):

    x = x.astype('float32')
    width = x.shape[0]
    height = x.shape[1]

    horizontal_num = 0
    vertical_num = 0

    horizontal_direction_last = 0
    horizontal_direction_current = 0

    vertical_direction_last = 0
    vertical_direction_current = 0

    for i in range(height):
        for j in range(width - 1):
            if((x[i, j + 1] - x[i, j]) > 0):
                horizontal_direction_current = 1
            if((x[i, j + 1] - x[i, j]) < 0):
                horizontal_direction_current = -1

            if(horizontal_direction_current * horizontal_direction_last < 0):
                horizontal_num += 1
            horizontal_direction_last = horizontal_direction_current
        break

    for i in range(height - 1):
        for j in range(width):
            if ((x[i + 1, j] - x[i, j]) > 0):
                vertical_direction_current = 1
            if((x[i + 1, j] - x[i, j]) < 0):
                vertical_direction_current = -1

            if(vertical_direction_current * vertical_direction_last < 0):
                vertical_num += 1
            vertical_direction_last = vertical_direction_current

    Total_rate = (horizontal_num + vertical_num) / (height * (width - 2) + (height - 2) * width)

    return Total_rate


