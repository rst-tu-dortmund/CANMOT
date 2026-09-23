#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;


// Point structure
struct Point {
    float x, y;
};

// Calculate determinant (cross product in 2D)
float cross_product(Point a, Point b, Point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// Check if a point is inside a convex polygon (sequence of points in order)
// This is not strictly needed if we use Sutherland-Hodgman clipping properly.

// Polygon area
float polygon_area(const std::vector<Point>& polygon) {
    if (polygon.size() < 3) return 0.0f;
    float area = 0.0f;
    for (size_t i = 0; i < polygon.size(); ++i) {
        size_t j = (i + 1) % polygon.size();
        area += polygon[i].x * polygon[j].y;
        area -= polygon[j].x * polygon[i].y;
    }
    return 0.5f * std::abs(area);
}

// Sutherland-Hodgman clipping
std::vector<Point> clip_polygon(const std::vector<Point>& subject_polygon, const std::vector<Point>& clip_polygon) {
    std::vector<Point> output_list = subject_polygon;

    for (size_t i = 0; i < clip_polygon.size(); ++i) {
        std::vector<Point> input_list = output_list;
        output_list.clear();

        if (input_list.empty()) break;

        Point clip_edge_start = clip_polygon[i];
        Point clip_edge_end = clip_polygon[(i + 1) % clip_polygon.size()];

        for (size_t j = 0; j < input_list.size(); ++j) {
            Point current_point = input_list[j];
            Point prev_point = input_list[(j + input_list.size() - 1) % input_list.size()];

            // Check if points are inside the clip edge
            // Assuming counter-clockwise winding for clip_polygon
            // cross_product > 0 means "left of edge" which is inside

            float cp_current = cross_product(clip_edge_start, clip_edge_end, current_point);
            float cp_prev = cross_product(clip_edge_start, clip_edge_end, prev_point);

            // To be robust, let's treat points on the line as inside (>= 0)
            // But floating point issues... let's say > -EPS
            const float EPS = 1e-7f;

            bool current_inside = cp_current >= -EPS;
            bool prev_inside = cp_prev >= -EPS;

            if (current_inside) {
                if (!prev_inside) {
                    // Intersection
                    float t = cp_prev / (cp_prev - cp_current);
                    Point intersection;
                    intersection.x = prev_point.x + t * (current_point.x - prev_point.x);
                    intersection.y = prev_point.y + t * (current_point.y - prev_point.y);
                    output_list.push_back(intersection);
                }
                output_list.push_back(current_point);
            } else if (prev_inside) {
                // Intersection
                float t = cp_prev / (cp_prev - cp_current);
                Point intersection;
                intersection.x = prev_point.x + t * (current_point.x - prev_point.x);
                intersection.y = prev_point.y + t * (current_point.y - prev_point.y);
                output_list.push_back(intersection);
            }
        }
    }
    return output_list;
}

// Convert box parameters to polygon (4 points)
std::vector<Point> get_box_polygon(float cx, float cy, float l, float w, float hdg) {
    float cos_h = std::cos(hdg);
    float sin_h = std::sin(hdg);

    // half dimensions
    float l2 = l / 2.0f;
    float w2 = w / 2.0f;

    // Corners relative to center
    // x1 =  l/2, -w/2  (This depends on coordinate system definition in python code)
    // The python code says:
    // x1 = center_x + length / 2 * cos_hdg - width / 2 * sin_hdg
    // y1 = center_y + length / 2 * sin_hdg + width / 2 * cos_hdg
    // This corresponds to rotating (l/2, w/2) ?

    /* Python code:
    x1 = center_x + length / 2 * cos_hdg - width / 2 * sin_hdg
    y1 = center_y + length / 2 * sin_hdg + width / 2 * cos_hdg
    x2 = center_x - length / 2 * cos_hdg - width / 2 * sin_hdg
    y2 = center_y - length / 2 * sin_hdg + width / 2 * cos_hdg
    x3 = center_x - length / 2 * cos_hdg + width / 2 * sin_hdg
    y3 = center_y - length / 2 * sin_hdg - width / 2 * cos_hdg
    x4 = center_x + length / 2 * cos_hdg + width / 2 * sin_hdg
    y4 = center_y + length / 2 * sin_hdg - width / 2 * cos_hdg
    */

    // Let's implement exactly as python
    Point p1, p2, p3, p4;

    // (l/2, w/2)
    p1.x = cx + l2 * cos_h - w2 * sin_h;
    p1.y = cy + l2 * sin_h + w2 * cos_h;

    // (-l/2, w/2)
    p2.x = cx - l2 * cos_h - w2 * sin_h;
    p2.y = cy - l2 * sin_h + w2 * cos_h;

    // (-l/2, -w/2)  <-- This order in python seems to correspond to p2, then...

    /*
      Checking Python order:
      x1, y1: (+l/2, +w/2)  (if we consider local axes aligned with heading)
      x2, y2: (-l/2, +w/2)
      x3, y3: (-l/2, -w/2)
      x4, y4: (+l/2, -w/2)

      This is Counter-Clockwise (CCW) if we assume standard Cartesian coords and rotation.
      ( + + ) -> ( - + ) -> ( - - ) -> ( + - )
      Q1 -> Q2 -> Q3 -> Q4. Correct.
    */

    // (-l/2, -w/2)
    p3.x = cx - l2 * cos_h + w2 * sin_h;
    p3.y = cy - l2 * sin_h - w2 * cos_h;

    // (l/2, -w/2)
    p4.x = cx + l2 * cos_h + w2 * sin_h;
    p4.y = cy + l2 * sin_h - w2 * cos_h;

    return {p1, p2, p3, p4};
}


py::array_t<float> rotated_iou_bev_cpp(
    py::array_t<float> boxes1,
    py::array_t<float> boxes2,
    py::array_t<bool> boxes1_mask,
    py::array_t<bool> boxes2_mask
) {
    // boxes1: (B, M, 7)    x, y, length, width, v_x, v_y, yaw
    // boxes2: (B, N, 7)
    // masks: (B, M) and (B, N) - boolean

    py::buffer_info buf1 = boxes1.request();
    py::buffer_info buf2 = boxes2.request();
    py::buffer_info mask1_buf = boxes1_mask.request();
    py::buffer_info mask2_buf = boxes2_mask.request();

    if (buf1.ndim != 3 || buf1.shape[2] != 7) {
        throw std::runtime_error("boxes1 must have shape (B, M, 7)");
    }
    if (buf2.ndim != 3 || buf2.shape[2] != 7) {
        throw std::runtime_error("boxes2 must have shape (B, N, 7)");
    }

    ssize_t B = buf1.shape[0];
    ssize_t M = buf1.shape[1];
    ssize_t N = buf2.shape[1];

    // Output array
    py::array_t<float> iou_matrix({B, M, N});

    // Accessors
    auto b1_acc = boxes1.unchecked<3>();
    auto b2_acc = boxes2.unchecked<3>();
    auto m1_acc = boxes1_mask.unchecked<2>();
    auto m2_acc = boxes2_mask.unchecked<2>();
    auto iou_acc = iou_matrix.mutable_unchecked<3>();

    // Parallelize over Batch and M (openmp)
    #pragma omp parallel for collapse(2)
    for (ssize_t b = 0; b < B; ++b) {
        for (ssize_t i = 0; i < M; ++i) {
            // Check mask 1
            if (mask1_buf.size > 0 && !m1_acc(b, i)) {
                 // Zero out remaining N
                 for (ssize_t j = 0; j < N; ++j) {
                     iou_acc(b, i, j) = 0.0f;
                 }
                continue;
            }

            // Get box 1 polygon
            std::vector<Point> poly1 = get_box_polygon(
                b1_acc(b, i, 0), b1_acc(b, i, 1), // cx, cy
                b1_acc(b, i, 2), b1_acc(b, i, 3), // l, w
                b1_acc(b, i, 6)                   // hdg
            );
            float area1 = b1_acc(b, i, 2) * b1_acc(b, i, 3); // l * w

            for (ssize_t j = 0; j < N; ++j) {
                // Check mask 2
                 if (mask2_buf.size > 0 && !m2_acc(b, j)) {
                    iou_acc(b, i, j) = 0.0f;
                    continue;
                }

                // Get box 2 polygon
                std::vector<Point> poly2 = get_box_polygon(
                    b2_acc(b, j, 0), b2_acc(b, j, 1),
                    b2_acc(b, j, 2), b2_acc(b, j, 3),
                    b2_acc(b, j, 6)
                );
                float area2 = b2_acc(b, j, 2) * b2_acc(b, j, 3);

                // Check intersection
                std::vector<Point> intersect_poly = clip_polygon(poly1, poly2);
                float intersect_area = polygon_area(intersect_poly);

                float union_area = area1 + area2 - intersect_area;

                if (union_area < 1e-4f) {
                    iou_acc(b, i, j) = 0.0f;
                } else {
                    iou_acc(b, i, j) = intersect_area / union_area;
                }
            }
        }
    }

    return iou_matrix;
}

PYBIND11_MODULE(iou_bev_cpp, m) {
    m.def("rotated_iou_bev_cpp", &rotated_iou_bev_cpp, "Rotated IoU BEV (CPP)");
}
