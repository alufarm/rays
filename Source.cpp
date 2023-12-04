#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <iostream>
#include <chrono>
//#include <limits>
#include <sstream>
#include <thread>
#include <vector>
#include <random>
#include <mutex>
using namespace std::chrono;

float randomf() {
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> dist(0, 1);
	return dist(engine);
}

float randomf(float min, float max) {
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> dist(min, max);
	return dist(engine);
}

class Timer {
public:
	Timer() : start(high_resolution_clock::now()) {}

	double elapsed() {
		double elapsed = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
		start = high_resolution_clock::now();
		return elapsed;
	}
private:
	high_resolution_clock::time_point start;
};

struct vec2 {
	float m_x, m_y;

	vec2() :m_x(0), m_y(0) {}
	vec2(float x, float y) :m_x(x), m_y(y) {}
	vec2(const vec2& v) :m_x(v.m_x), m_y(v.m_y) {}

	vec2& operator=(vec2 v) {
		m_x = v.m_x;
		m_y = v.m_y;
		return *this;
	}

	vec2 operator+(const vec2& v) const {
		return vec2(m_x + v.m_x, m_y + v.m_y);
	}

	vec2 operator-(const vec2& v) const {
		return vec2(m_x - v.m_x, m_y - v.m_y);
	}

	vec2 operator+(const float& v) const {
		return vec2(m_x + v, m_y + v);
	}

	vec2 operator-(const float& v) const {
		return vec2(m_x - v, m_y - v);
	}

	vec2 operator*(const float& v) const {
		return vec2(m_x * v, m_y * v);
	}

	vec2 operator/(const float& v) const {
		return vec2(m_x / v, m_y / v);
	}

	vec2& operator+=(const vec2& v) {
		m_x += v.m_x;
		m_y += v.m_y;
		return *this;
	}

	vec2& operator-=(const vec2& v) {
		m_x -= v.m_x;
		m_y -= v.m_y;
		return *this;
	}

	vec2& operator+=(const float& v) {
		m_x += v;
		m_y += v;
		return *this;
	}

	vec2& operator-=(const float& v) {
		m_x -= v;
		m_y -= v;
		return *this;
	}

	vec2& operator*=(const float& v) {
		m_x *= v;
		m_y *= v;
		return *this;
	}

	vec2& operator/=(const float& v) {
		m_x /= v;
		m_y /= v;
		return *this;
	}

	float Magnitude() {
		return sqrt(m_x * m_x + m_y * m_y);
	}

	void Normalize() {
		if (m_x == 0 || m_y == 0) return;
		float m = Magnitude();
		m_x /= m;
		m_y /= m;
	}
};

struct vec3 {
	float m_x, m_y, m_z;

	vec3() :m_x(0), m_y(0), m_z(0) {}
	vec3(float x, float y, float z) :m_x(x), m_y(y), m_z(z) {}
	vec3(const vec3& v) :m_x(v.m_x), m_y(v.m_y), m_z(v.m_z) {}

	vec3& operator=(vec3 v) {
		m_x = v.m_x;
		m_y = v.m_y;
		m_z = v.m_z;
		return *this;
	}

	vec3 operator+(const vec3& v) const {
		return vec3(m_x + v.m_x, m_y + v.m_y, m_z + v.m_z);
	}

	vec3 operator-(const vec3& v) const {
		return vec3(m_x - v.m_x, m_y - v.m_y, m_z - v.m_z);
	}

	vec3 operator+(const float& v) const {
		return vec3(m_x + v, m_y + v, m_z + v);
	}

	vec3 operator-(const float& v) const {
		return vec3(m_x - v, m_y - v, m_z - v);
	}

	vec3 operator*(const float& v) const {
		return vec3(m_x * v, m_y * v, m_z * v);
	}

	vec3 operator/(const float& v) const {
		return vec3(m_x / v, m_y / v, m_z / v);
	}

	vec3& operator+=(const vec3& v) {
		m_x += v.m_x;
		m_y += v.m_y;
		m_z += v.m_z;
		return *this;
	}

	vec3& operator-=(const vec3& v) {
		m_x -= v.m_x;
		m_y -= v.m_y;
		m_z -= v.m_z;
		return *this;
	}

	vec3& operator+=(const float& v) {
		m_x += v;
		m_y += v;
		m_z += v;
		return *this;
	}

	vec3& operator-=(const float& v) {
		m_x -= v;
		m_y -= v;
		m_z -= v;
		return *this;
	}

	vec3& operator*=(const float& v) {
		m_x *= v;
		m_y *= v;
		m_z *= v;
		return *this;
	}

	vec3& operator/=(const float& v) {
		m_x /= v;
		m_y /= v;
		m_z /= v;
		return *this;
	}

	float Magnitude() {
		return sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
	}

	void Normalize() {
		float m = Magnitude();
		if (m == 0)return;
		m_x /= m;
		m_y /= m;
		m_z /= m;
	}

	float Dot(vec3 v) {
		return m_x * v.m_x + m_y * v.m_y + m_z * v.m_z;
	}

	vec3 Cross(vec3& v) {
		return vec3(m_y * v.m_z - m_z * v.m_y,
			m_z * v.m_x - m_x * v.m_z,
			m_x * v.m_y - m_y * v.m_x);
	}
};

vec2 GetSize(HWND hWnd) {
	RECT rc;
	GetWindowRect(hWnd, &rc);
	return vec2(rc.right - rc.left, rc.bottom - rc.top);
}

void Rectangle(HDC hDc, vec2 pos, vec2 size, COLORREF color) {
	HPEN pen = CreatePen(PS_SOLID, 1, color);
	HBRUSH brush = CreateSolidBrush(color);
	SelectObject(hDc, pen);
	SelectObject(hDc, brush);
	Rectangle(hDc, pos.m_x, pos.m_y, pos.m_x + size.m_x, pos.m_y + size.m_y);
	DeleteObject(pen);
	DeleteObject(brush);
}

void SetPixel(HDC hDc, vec2 pos, COLORREF color) {
	HPEN pen = CreatePen(PS_SOLID, 1, color);
	HBRUSH brush = CreateSolidBrush(color);
	SelectObject(hDc, pen);
	SelectObject(hDc, brush);
	SetPixel(hDc, pos.m_x, pos.m_y, color);
	DeleteObject(pen);
	DeleteObject(brush);
}

bool isKeyPressed(int key) {
	return GetAsyncKeyState(key);
}

bool intersect(vec3 center, vec3 pos, float r) {
	vec3 v(center - pos);
	v.Normalize();
	float x = v.Dot(center);
	v = v * x - center;
	if (v.Dot(v) > r * r)return false;
	return true;
}

struct Ray {
	vec3 origin;
	vec3 dir;
};

struct Triangle {
	vec3 vertices[3];
	vec3 normal;

	Triangle(vec3 a, vec3 b, vec3 c) {
		vertices[0] = a;
		vertices[1] = b;
		vertices[2] = c;
	}

	void calcNormal() {
		vec3 e1 = vertices[1] - vertices[0];
		vec3 e2 = vertices[2] - vertices[0];
		normal = e1.Cross(e2);
		normal.Normalize();
	}
};

bool intersectTriangleRay(Ray& ray, Triangle& triangle) {
	vec3 e1 = triangle.vertices[1] - triangle.vertices[0];
	vec3 e2 = triangle.vertices[2] - triangle.vertices[0];
	vec3 pvec = ray.dir.Cross(e2);
	float det = e1.Dot(pvec);
	float epsilon = 1e-8;
	if (det < epsilon && det > -epsilon)return false;

	float inv = 1 / det;
	vec3 tvec = ray.origin - triangle.vertices[0];
	float u = tvec.Dot(pvec) * inv;

	if (u < 0 || u > 1) return false;

	vec3 qvec = tvec.Cross(e1);
	float v = ray.dir.Dot(qvec) * inv;
	if (v < 0 || u + v >1)return false;
	float t = e2.Dot(qvec) * inv;
	return true;
}

#define PI 3.14f
void sphereGenerate(vec3 origin, float r) {

	//for (int i = 0; i < 2 * PI; i+=0.1f) {
	//	float x = cos(i) * r;
	//	float y = sin(i) * r;
	//	float z = 
	//}
}

int main() {

	HWND hWnd = GetConsoleWindow();
	HDC hDc = GetDC(hWnd);
	HDC hDcBuf = CreateCompatibleDC(hDc);

	vec2 wsize = GetSize(hWnd);
	float aspect = wsize.m_x / wsize.m_y;

	vec2 size(20, 20);
	vec2 pos(wsize / 2 - size / 2);
	vec3 color(0, 1, 0);
	float speed = 500;

	std::vector<Triangle> cube = {
		Triangle(vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(1, 1, -1)),
		Triangle(vec3(-1, -1, -1), vec3(1, 1, -1), vec3(1, -1, -1))
	};

	for (Triangle& i : cube) {
		i.calcNormal();
	}

	vec3 lightPos = vec3(0, 0, 1);
	Ray r;

	std::vector<std::thread*> pool;
	int num_threads = 2;
	std::vector<vec3> thread_colors;
	for (int i = 0; i < num_threads; i++) {
		float r = randomf(0.5, 1);
		float g = randomf(0.5, 1);
		float b = randomf(0.5, 1);
		vec3 col(r, g, b);
		thread_colors.push_back(col);
	}

	Timer perf_timer;
	double perf = 0;

	std::mutex mutex;

	BITMAPINFO BitMapInfo;
	BitMapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	BitMapInfo.bmiHeader.biWidth = wsize.m_x;
	BitMapInfo.bmiHeader.biHeight = wsize.m_y;
	BitMapInfo.bmiHeader.biPlanes = 1;
	BitMapInfo.bmiHeader.biBitCount = 32;
	BitMapInfo.bmiHeader.biCompression = BI_RGB;

	void* buffer = new unsigned int[wsize.m_x * wsize.m_y * sizeof(unsigned int)];
	unsigned int* pixels;

	float k = 0;
	float delta;
	Timer timer;
	while (true) {

		delta = timer.elapsed();
		std::string str;
		std::stringstream ss;
		ss << "FPS: " << 1 / delta << " Frametime: " << delta * 1000 << " Threads: " << pool.size() << " Objects time: " << perf * 1000;
		str = ss.str();
		std::wstring title = std::wstring(str.begin(), str.end());
		SetWindowText(hWnd, title.c_str());

		/*vec2 input(isKeyPressed(VK_RIGHT) - isKeyPressed(VK_LEFT), isKeyPressed(VK_DOWN) - isKeyPressed(VK_UP));
		input.Normalize();
		pos += input * speed * delta;*/

		//HBITMAP bitmap = CreateCompatibleBitmap(hDc, wsize.m_x, wsize.m_y);
		//SelectObject(hDcBuf, bitmap);

		k += 1 * delta;
		lightPos.m_x = sin(k) * 2;

		if (pool.empty() == false) {
			for (auto& item : pool) {
				delete item;
			}
			pool.clear();
		}

		//for (int i = 0; i < num_threads; i++) {
		//	//vec3 color = thread_colors[i];
		//	std::thread* t = new std::thread(
		//		[&]() {
		//			for (size_t x = 0; x < wsize.m_x; x++) {
		//				for (size_t y = i * (wsize.m_y / num_threads); y < i * (wsize.m_y / num_threads) + wsize.m_y / num_threads; y++) {
		//					float rx = (x * 2) / wsize.m_x - 1;
		//					float ry = (y * 2) / wsize.m_y - 1;
		//					rx *= aspect;

		//					Ray rr;
		//					rr.origin = vec3(0, 0, 1);
		//					rr.dir = vec3(rx, ry, 0) - r.origin;
		//					rr.dir.Normalize();

		//					for (Triangle& i : cube) {
		//						if (intersectTriangleRay(rr, i)) {
		//							vec3 lightDir = vec3(rx, ry, 0) - lightPos;
		//							lightDir.Normalize();
		//							float diff = max(lightDir.Dot(i.normal), 0);
		//							vec3 diffuse = vec3(1.0f, 1.0f, 1.0f) * diff;
		//							vec3 result = vec3(diffuse.m_x * color.m_x, diffuse.m_y * color.m_y, diffuse.m_z * color.m_z);

		//							mutex.lock();
		//							SetPixel(hDcBuf, x, y, RGB((int)(result.m_x * 255), (int)(result.m_y * 255), (int)(result.m_z * 255)));
		//							mutex.unlock();
		//						}
		//					}
		//				}
		//			}

		//		}
		//	);

		//	pool.push_back(t);
		//}

		//for (int i = 0; i < num_threads; i++) {
		//	pool[i]->join();
		//}

		/*pixels = (unsigned int*)buffer;
		for (int i = 0; i < size.m_x * size.m_y; i++) {
			*pixels = RGB(255, 0, 0);
			pixels++;
		}*/

		//for (int i = 0; i < num_threads; i++) {
		//	color = thread_colors[i];
		//	std::thread* t = new std::thread(
		//		[&]() {
		//			unsigned int* subbuf = new unsigned int[wsize.m_x * (wsize.m_y / num_threads) * sizeof(unsigned int)];
		//			unsigned int* px = subbuf;

		//			for (size_t x = 0; x < wsize.m_x; x++) {
		//				for (size_t y = 0; y < wsize.m_y / num_threads; y++) {
		//					px[y * (int)wsize.m_x + x] = RGB((int)(color.m_z * 255), (int)(color.m_y * 255), (int)(color.m_x * 255));
		//				}
		//			}

		//			/*mutex.lock();
		//			int l = 0;
		//			for (int k = i * (wsize.m_y / num_threads); k < i * (wsize.m_y / num_threads) + wsize.m_y / num_threads; k++) {
		//				pixels[k] = px[l];
		//				l++;
		//			}
		//			mutex.unlock();*/

		//			delete [] subbuf;
		//		}
		//	);

		//	pool.push_back(t);
		//}

		//for (int i = 0; i < num_threads; i++) {
		//	pool[i]->join();
		//}

		/*StretchDIBits(hDc,
			0, 0, wsize.m_x, wsize.m_y,
			0, 0, wsize.m_x, wsize.m_y,
			buffer,
			&BitMapInfo,
			DIB_RGB_COLORS,
			SRCCOPY);*/

		//BitBlt(hDc, 0, 0, wsize.m_x, wsize.m_y, hDcBuf, 0, 0, SRCCOPY);
		//DeleteObject(bitmap);
	}

	return 0;
}