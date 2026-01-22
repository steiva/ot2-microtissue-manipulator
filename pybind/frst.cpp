#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;


// ============================================================
// Convert any numeric numpy array to double buffer
// ============================================================

void to_double(const py::array &arr, std::vector<double> &dst)
{
    py::buffer_info info = arr.request();
    auto H = info.shape[0];
    auto W = info.shape[1];
    dst.resize(H * W);

    const char *src = static_cast<const char *>(info.ptr);
    size_t s0 = info.strides[0];
    size_t s1 = info.strides[1];

    // dtype size
    size_t itemsize = info.itemsize;

    for (py::ssize_t y = 0; y < H; y++) {
        for (py::ssize_t x = 0; x < W; x++) {
            const char *p = src + y * s0 + x * s1;

            // Interpret element based on dtype
            switch (info.format[0]) {
                case 'd': dst[y * W + x] = *reinterpret_cast<const double *>(p); break;
                case 'f': dst[y * W + x] = *reinterpret_cast<const float *>(p); break;
                case 'i': dst[y * W + x] = static_cast<double>(*reinterpret_cast<const int *>(p)); break;
                case 'I': dst[y * W + x] = static_cast<double>(*reinterpret_cast<const unsigned int *>(p)); break;
                case 'h': dst[y * W + x] = static_cast<double>(*reinterpret_cast<const short *>(p)); break;
                case 'H': dst[y * W + x] = static_cast<double>(*reinterpret_cast<const unsigned short *>(p)); break;
                case 'b': dst[y * W + x] = static_cast<double>(*reinterpret_cast<const int8_t *>(p)); break;
                case 'B': dst[y * W + x] = static_cast<double>(*reinterpret_cast<const uint8_t *>(p)); break;
                case 'l': dst[y * W + x] = static_cast<double>(*reinterpret_cast<const long *>(p)); break;
                case 'L': dst[y * W + x] = static_cast<double>(*reinterpret_cast<const unsigned long *>(p)); break;
                default:
                    throw std::runtime_error("Unsupported dtype.");
            }
        }
    }
}


// ============================================================
// Sobel gradients
// ============================================================

void sobel(const std::vector<double> &img, py::ssize_t H, py::ssize_t W,
           std::vector<double> &gx, std::vector<double> &gy, std::vector<double> &mag)
{
    gx.assign(H*W, 0.0);
    gy.assign(H*W, 0.0);
    mag.assign(H*W, 0.0);

    static constexpr int Kx[3][3] = {
        {-1,0,1},{-2,0,2},{-1,0,1}
    };
    static constexpr int Ky[3][3] = {
        {-1,-2,-1},{0,0,0},{1,2,1}
    };

    for (py::ssize_t y = 1; y < H-1; y++) {
        for (py::ssize_t x = 1; x < W-1; x++) {
            double sx = 0.0, sy = 0.0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    double v = img[(y + ky)*W + (x + kx)];
                    sx += v * Kx[ky+1][kx+1];
                    sy += v * Ky[ky+1][kx+1];
                }
            }
            py::ssize_t idx = y*W + x;
            gx[idx] = sx;
            gy[idx] = sy;
            mag[idx] = std::sqrt(sx*sx + sy*sy);
        }
    }
}


// ============================================================
// Gaussian blur (separable)
// ============================================================

std::vector<double> gaussian_kernel(double sigma)
{
    int radius = std::max(1, int(std::ceil(3 * sigma)));
    int size = radius * 2 + 1;

    std::vector<double> ker(size);
    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        double v = std::exp(-(i*i) / (2 * sigma * sigma));
        ker[i + radius] = v;
        sum += v;
    }
    for (auto &v : ker) v /= sum;

    return ker;
}

void gaussian_blur(std::vector<double> &img, py::ssize_t H, py::ssize_t W, double sigma)
{
    auto ker = gaussian_kernel(sigma);
    int R = (ker.size() - 1) / 2;

    std::vector<double> tmp(H * W, 0.0);

    // Horizontal
    for (py::ssize_t y = 0; y < H; y++) {
        for (py::ssize_t x = 0; x < W; x++) {
            double s = 0.0;
            for (int k = -R; k <= R; k++) {
                py::ssize_t xx = std::clamp<py::ssize_t>(x + k, 0, W - 1);
                s += img[y*W + xx] * ker[k + R];
            }
            tmp[y*W + x] = s;
        }
    }

    // Vertical
    for (py::ssize_t y = 0; y < H; y++) {
        for (py::ssize_t x = 0; x < W; x++) {
            double s = 0.0;
            for (int k = -R; k <= R; k++) {
                py::ssize_t yy = std::clamp<py::ssize_t>(y + k, 0, H - 1);
                s += tmp[yy*W + x] * ker[k + R];
            }
            img[y*W + x] = s;
        }
    }
}


// ============================================================
// FRST accumulation for a radius
// ============================================================

void accumulate(const std::vector<double> &gx,
                const std::vector<double> &gy,
                const std::vector<double> &mag,
                py::ssize_t H, py::ssize_t W,
                int R, double beta,
                std::vector<double> &O,
                std::vector<double> &M)
{
    O.assign(H*W, 0.0);
    M.assign(H*W, 0.0);

    for (py::ssize_t y = 1; y < H-1; y++) {
        for (py::ssize_t x = 1; x < W-1; x++) {

            py::ssize_t idx = y*W + x;

            double m = mag[idx];
            if (m < beta) continue;

            double dx = gx[idx] * R;
            double dy = gy[idx] * R;

            int px = int(x + dx);
            int py = int(y + dy);

            if (0 <= px && px < W && 0 <= py && py < H) {
                py::ssize_t id = py*W + px;
                O[id] += 1.0;
                M[id] += m;
            }

            px = int(x - dx);
            py = int(y - dy);

            if (0 <= px && px < W && 0 <= py && py < H) {
                py::ssize_t id = py*W + px;
                O[id] += 1.0;
                M[id] += m;
            }
        }
    }

    double sigma = R * 0.25;
    gaussian_blur(O, H, W, sigma);
    gaussian_blur(M, H, W, sigma);
}


// ============================================================
// Main FRST
// ============================================================

py::array_t<double> frst(py::array image,
                         std::vector<int> radii,
                         double alpha)
{
    py::buffer_info info = image.request();
    py::ssize_t H = info.shape[0];
    py::ssize_t W = info.shape[1];

    std::vector<double> img;
    to_double(image, img);

    std::vector<double> gx, gy, mag;
    sobel(img, H, W, gx, gy, mag);

    // normalize gradients
    for (py::ssize_t i = 0; i < H*W; i++) {
        double m = mag[i] + 1e-12;
        gx[i] /= m;
        gy[i] /= m;
    }

    std::vector<double> acc(H*W, 0.0);
    std::vector<double> O, M;

    for (int r : radii) {
        accumulate(gx, gy, mag, H, W, r, 0.5, O, M);
        for (py::ssize_t i = 0; i < H*W; i++) {
            acc[i] += std::pow(O[i], alpha) * M[i];
        }
    }

    // normalize to 0-255
    double mn = *std::min_element(acc.begin(), acc.end());
    double mx = *std::max_element(acc.begin(), acc.end());
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 1.0;

    py::array_t<double> out({H, W});
    auto outbuf = out.mutable_unchecked<2>();

    for (py::ssize_t y = 0; y < H; y++) {
        for (py::ssize_t x = 0; x < W; x++) {
            outbuf(y, x) = (acc[y*W + x] - mn) * scale;
        }
    }

    return out;
}


// ============================================================
// PYBIND11 MODULE
// ============================================================

PYBIND11_MODULE(frst, m)
{
    m.def("frst", &frst, "Fast Radial Symmetry Transform");
}
