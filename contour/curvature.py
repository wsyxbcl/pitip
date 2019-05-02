import matplotlib.pyplot as plt

def find_circle(b, c, d):
    """
    Use 3 points(a, b and c) to fix a circle.
    """
    temp = c[0]**2 + c[1]**2
    bc = (b[0]**2 + b[1]**2 - temp) / 2
    cd = (temp - d[0]**2 - d[1]**2) / 2
    det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

    if abs(det) < 1.0e-10:
        raise ValueError('Points on same line')

    # Center of circle
    center_x = (bc*(c[1] - d[1]) - cd*(b[1] - c[1])) / det
    center_y = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

    radius = ((center_x - b[0])**2 + (center_y - b[1])**2)**.5

    return center_x, center_y, radius

def cal_curvature(x, y, cyc=1, interval=1, sign=1):
    """
    Calculate the curvature of each point of a given curve.
    At each point, we calculate the curvature by fitting a circle to that
    point and the two points that are #interval points away from it. 

    The curvature is then defined as the reciprocal of the radius of that circle

    Args
        x, y: lists that contains coordinates of points on the curve
        cyc: cyc=1 if the curve in in a cyclic form
        interval: number of points away from the middle point and the other two points
                  for example, interval = 2 -> . . . x . x . x . . .   
        sign: whether to distinguish the sign of the curvature
        
    Returns
        cv: list of curvature at each point
    """
    cv = []
    for i in range(len(x)):
        try: 
            if cyc:
                x_midx, x_lidx, x_hidx = x[i], x[i-interval], x[(i+interval)%len(x)]
                y_midx, y_lidx, y_hidx = y[i], y[i-interval], y[(i+interval)%len(y)]
            else:
                if i - interval < 0:
                    x_midx, x_lidx, x_hidx = x[i], x[i], x[i+interval]
                    y_midx, y_lidx, y_hidx = y[i], y[i], y[i+interval]
                elif i + interval > (len(x) - 1):
                    x_midx, x_lidx, x_hidx = x[i], x[i-interval], x[i]
                    y_midx, y_lidx, y_hidx = y[i], y[i-interval], y[i]
                else:
                    x_midx, x_lidx, x_hidx = x[i], x[i-interval], x[i+interval]
                    y_midx, y_lidx, y_hidx = y[i], y[i-interval], y[i+interval]    
        except IndexError:
            print("interval value too large for the number of points")
            raise
            
        if sign:
            s = (x_lidx - x_midx) * (y_hidx - y_midx) - (y_lidx - y_midx) * (x_hidx - x_midx)
        else:
            s = 1

        try:
            center_x, center_y, radius = find_circle([x_midx, y_midx],
                                                 [x_lidx, y_lidx],
                                                 [x_hidx, y_hidx])
        except ValueError:
            cv.append(0.0)
            continue
            
        if s > 0:
            cv.append(1.0/-radius)
        else:
            cv.append(1.0/radius)
    return cv


if __name__ == '__main__':
    a = [1, 2]
    b = [2, 2]
    c = [3, 5]

    center_x, center_y, radius = find_circle(a, b, c)
    circle = plt.Circle((center_x, center_y), radius, color = 'r', fill=False)
    # circle = plt.Circle((0, 0), 1, color = 'r', fill=False)

    fig, ax = plt.subplots()

    ax.add_artist(circle)
    ax.plot(a[0], a[1], 'o')
    ax.plot(b[0], b[1], 'o')
    ax.plot(c[0], c[1], 'o')
    plt.show()
    cv = cal_curvature([1,2,3], [2,2,5])