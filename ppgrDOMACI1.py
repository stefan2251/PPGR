import cv2
import numpy as np

def homogenize(pt):
    return np.array([pt[0], pt[1], 1])

def dehomogenize(v):
    return np.array([v[0]/v[2], v[1]/v[2]])

def line_through(a, b):
    return np.cross(a, b)

def intersection(l1, l2):
    p = np.cross(l1, l2)
    return dehomogenize(p)


def eighth_point(v):
    p1, p2, p3, p5, p6, p7, p8 = v

    xb1 = intersection(line_through(p2, p6), line_through(p1, p5))
    xb2 = intersection(line_through(p2, p6), line_through(p7, p3))
    xb3 = intersection(line_through(p7, p3), line_through(p5, p1))
    xb = np.mean([xb1, xb2, xb3], axis=0)

    yb1 = intersection(line_through(p6, p5), line_through(p7, p8))
    yb2 = intersection(line_through(p2, p1), line_through(p6, p5))
    yb3 = intersection(line_through(p7, p8), line_through(p2, p1))
    yb = np.mean([yb1, yb2, yb3], axis=0)

    zb1 = intersection(line_through(p2, p3), line_through(p6, p7))
    zb2 = intersection(line_through(p5, p8), line_through(p6, p7))
    zb3 = intersection(line_through(p2, p3), line_through(p5, p8))
    zb = np.mean([zb1, zb2, zb3], axis=0)

    p41 = intersection(line_through(homogenize(xb), p8), line_through(homogenize(yb), p3))
    p42 = intersection(line_through(homogenize(xb), p8), line_through(homogenize(zb), p1))
    p43 = intersection(line_through(homogenize(yb), p3), line_through(homogenize(zb), p1))

    p4 = np.mean([p41, p42, p43], axis=0)

    return int(p4[0]), int(p4[1])

def put_point(img, x, y, idx):
    cv2.circle(img, (x, y), 6, (255, 255, 255), -1)
    cv2.putText(img, f"P{idx}", (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def mouse_handler(event, x, y, flags, userdata):
    global points, image, click_index

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append(homogenize([x, y]))

        put_point(image, x, y, click_index)
        click_index += 1

        if click_index == 4:
            click_index += 1

        if len(points) == 7:
            px, py = eighth_point(points)
            put_point(image, px, py, 4)
            print(px, py)
            cv2.imshow("Find Missing Vertex", image)

if __name__ == "__main__":
    image = cv2.imread("osmoTemeSlika_25.png")
    cv2.imshow("Find Missing Vertex", image)

    points = []
    click_index = 1

    cv2.setMouseCallback("Find Missing Vertex", mouse_handler)
    cv2.waitKey(0)
    cv2.destroyAllWindows()