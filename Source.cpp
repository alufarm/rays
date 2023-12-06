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

#define PI 3.14f

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

	template<typename T>
	vec3& operator=(T v) {
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

struct vec4 {
	float m_x, m_y, m_z, m_w;

	vec4() :m_x(0), m_y(0), m_z(0), m_w(0) {}
	vec4(float x, float y, float z, float w) :m_x(x), m_y(y), m_z(z), m_w(w) {}
	vec4(const vec4& v) :m_x(v.m_x), m_y(v.m_y), m_z(v.m_z), m_w(v.m_w) {}
	vec4(const vec3& v) :m_x(v.m_x), m_y(v.m_y), m_z(v.m_z), m_w(0) {}
	vec4(const vec3& v, float w) :m_x(v.m_x), m_y(v.m_y), m_z(v.m_z), m_w(w) {}

	vec4& operator=(vec4 v) {
		m_x = v.m_x;
		m_y = v.m_y;
		m_z = v.m_z;
		m_w = v.m_w;
		return *this;
	}

	vec4& operator=(vec3 v) {
		m_x = v.m_x;
		m_y = v.m_y;
		m_z = v.m_z;
		m_w = 0;
		return *this;
	}

	vec4 operator+(const vec4& v) const {
		return vec4(m_x + v.m_x, m_y + v.m_y, m_z + v.m_z, m_w + v.m_w);
	}

	vec4 operator-(const vec4& v) const {
		return vec4(m_x - v.m_x, m_y - v.m_y, m_z - v.m_z, m_w - v.m_w);
	}

	vec4 operator+(const float& v) const {
		return vec4(m_x + v, m_y + v, m_z + v, m_w + v);
	}

	vec4 operator-(const float& v) const {
		return vec4(m_x - v, m_y - v, m_z - v, m_w - v);
	}

	vec4 operator*(const float& v) const {
		return vec4(m_x * v, m_y * v, m_z * v, m_w * v);
	}

	vec4 operator/(const float& v) const {
		return vec4(m_x / v, m_y / v, m_z / v, m_w / v);
	}

	vec4& operator+=(const vec4& v) {
		m_x += v.m_x;
		m_y += v.m_y;
		m_z += v.m_z;
		m_w += v.m_w;
		return *this;
	}

	vec4& operator-=(const vec4& v) {
		m_x -= v.m_x;
		m_y -= v.m_y;
		m_z -= v.m_z;
		m_w -= v.m_w;
		return *this;
	}

	vec4& operator+=(const float& v) {
		m_x += v;
		m_y += v;
		m_z += v;
		m_w += v;
		return *this;
	}

	vec4& operator-=(const float& v) {
		m_x -= v;
		m_y -= v;
		m_z -= v;
		m_w -= v;
		return *this;
	}

	vec4& operator*=(const float& v) {
		m_x *= v;
		m_y *= v;
		m_z *= v;
		m_w *= v;
		return *this;
	}

	vec4& operator/=(const float& v) {
		m_x /= v;
		m_y /= v;
		m_z /= v;
		m_w /= v;
		return *this;
	}

	float Magnitude() {
		return sqrt(m_x * m_x + m_y * m_y + m_z * m_z + m_w * m_w);
	}

	void Normalize() {
		float m = Magnitude();
		if (m == 0)return;
		m_x /= m;
		m_y /= m;
		m_z /= m;
		m_w /= m;
	}
};

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
	std::vector<vec3> vertices;
	vec3 normal;
	Triangle(vec3 a, vec3 b, vec3 c) {
		vertices.push_back(a);
		vertices.push_back(b);
		vertices.push_back(c);
	}

	Triangle(vec3 a, vec3 b, vec3 c, vec3 normal) : normal(normal) {
		vertices.push_back(a);
		vertices.push_back(b);
		vertices.push_back(c);
	}

	void calcNormal() {
		vec3 e1 = vertices[1] - vertices[0];
		vec3 e2 = vertices[2] - vertices[0];
		normal = e2.Cross(e1);
		normal.Normalize();
	}
};

struct Mesh {
	std::vector<Triangle> triangles;

	Mesh() {
	}
};

bool intersectTriangleRay(Ray& ray, Triangle& triangle, float& t) {
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
	t = e2.Dot(qvec) * inv;
	return true;
}

class FrameBuffer {
public:
	int width, height;
	FrameBuffer(int width, int height) : width(width), height(height) {
		buffer = new unsigned int[width * height * sizeof(unsigned int)];
		pixel = (unsigned int*)buffer;

		BitMapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		BitMapInfo.bmiHeader.biWidth = width;
		BitMapInfo.bmiHeader.biHeight = height;
		BitMapInfo.bmiHeader.biPlanes = 1;
		BitMapInfo.bmiHeader.biBitCount = 32;
		BitMapInfo.bmiHeader.biCompression = BI_RGB;
	}

	void Clear(vec3 color) {
		color *= 255;
		for (int i = 0; i < width * height * sizeof(unsigned int); i++) {
			pixel[i] = RGB(color.m_z, color.m_y, color.m_x);
		}
	}

	void Clear() {
		memset(buffer, 0, width * height * sizeof(unsigned int));
	}

	void SetPixel(int x, int y, vec3 color) {
		color *= 255;
		pixel[y * width + x] = RGB(color.m_z, color.m_y, color.m_x);
	}

	void* GetPtr() {
		return buffer;
	}

	BITMAPINFO& GetBitmapInfo() {
		return BitMapInfo;
	};
private:
	void* buffer;
	unsigned int* pixel;
	BITMAPINFO BitMapInfo;
};

/*
	0 1 2 3
	4 5 6 7
	8 9 10 11
	12 13 14 15


*/
std::vector<float> mul4x4(std::vector<float>& a, std::vector<float>& b) {
	std::vector<float> c = {
		a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
		a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
		a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
		a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],

		a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
		a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
		a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
		a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],

		a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
		a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
		a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
		a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],

		a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
		a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
		a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
		a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15],
	};
	return c;
}

vec4 mul4x4(std::vector<float>& a, vec4& v) {
	vec4 c = {
		a[0] * v.m_x + a[1] * v.m_y + a[2] * v.m_z + a[3] * v.m_w,
		a[4] * v.m_x + a[5] * v.m_y + a[6] * v.m_z + a[7] * v.m_w,
		a[8] * v.m_x + a[9] * v.m_y + a[10] * v.m_z + a[11] * v.m_w,
		a[12] * v.m_x + a[13] * v.m_y + a[14] * v.m_z + a[15] * v.m_w,
	};
	return c;
}

std::vector<float> RotateY(std::vector<float>& mat, float angle) {
	std::vector<float> rotY = {
		cos(angle),0,sin(angle),0,
		0,1,0,0,
		-sin(angle),0,cos(angle),0,
		0,0,0,1
	};
	return mul4x4(mat, rotY);
}

std::vector<float> RotateX(std::vector<float>& mat, float angle) {
	std::vector<float> rotX = {
		1,0,0,0,
		0,cos(angle),-sin(angle),0,
		0,sin(angle),cos(angle),0,
		0,0,0,1
	};
	return mul4x4(mat, rotX);
}

std::vector<float> Translate(std::vector<float>& mat, vec3 to) {
	std::vector<float> translate = {
	1,0,0,to.m_x,
	0,1,0,to.m_y,
	0,0,1,to.m_z,
	0,0,0,1
	};
	return mul4x4(mat, translate);
}

void Identity(std::vector<float>& mat) {
	mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
}

class Window {
public:
	Window(unsigned int width, unsigned int height):width(width), height(height) {
		hWnd = GetConsoleWindow();
		hDc = GetDC(hWnd);
		SetWindowPos(hWnd, NULL, 100, 100, width, height, SWP_SHOWWINDOW);
	}

	vec2 GetSize() {
		return vec2(width, height);
	}

	void SetTitle(std::string& title) {
		std::wstring wtitle = std::wstring(title.begin(), title.end());
		SetWindowText(hWnd, wtitle.c_str());
	}

	void Render(FrameBuffer& frameBuffer) {
		StretchDIBits(hDc,
			0, 0, width, height,
			0, 0, width, height,
			frameBuffer.GetPtr(),
			&frameBuffer.GetBitmapInfo(),
			DIB_RGB_COLORS,
			SRCCOPY);
	}
private:
	unsigned int width, height;
	HWND hWnd;
	HDC hDc;
};

int main() {

	unsigned int width = 800, height = 600;
	vec2 wsize(width, height);
	Window window(width, height);

	float aspect = wsize.m_x / wsize.m_y;

	vec3 color(0, 1, 0);

	Mesh cube;
	cube.triangles = {
		Triangle(vec3(-0.5f, -0.5f, -0.5f),
		vec3(0.5f, -0.5f, -0.5f),
		vec3(0.5f,  0.5f, -0.5f)),
		Triangle(vec3(0.5f,  0.5f, -0.5f),
		vec3(-0.5f,  0.5f, -0.5f),
		vec3(-0.5f, -0.5f, -0.5f)),

		Triangle(vec3(-0.5f, -0.5f,  0.5f),
		vec3(0.5f, -0.5f,  0.5f),
		vec3(0.5f,  0.5f,  0.5f)),
		Triangle(vec3(0.5f,  0.5f,  0.5f),
		vec3(-0.5f,  0.5f,  0.5f),
		vec3(-0.5f, -0.5f,  0.5f)),

		Triangle(vec3(-0.5f,  0.5f,  0.5f),
		vec3(-0.5f,  0.5f, -0.5f),
		vec3(-0.5f, -0.5f, -0.5f)),
		Triangle(vec3(-0.5f, -0.5f, -0.5f),
		vec3(-0.5f, -0.5f,  0.5f),
		vec3(-0.5f,  0.5f,  0.5f)),

		Triangle(vec3(0.5f,  0.5f,  0.5f),
		vec3(0.5f,  0.5f, -0.5f),
		vec3(0.5f, -0.5f, -0.5f)),
		Triangle(vec3(0.5f, -0.5f, -0.5f),
		vec3(0.5f, -0.5f,  0.5f),
		vec3(0.5f,  0.5f,  0.5f)),

		Triangle(vec3(-0.5f, -0.5f, -0.5f),
		vec3(0.5f, -0.5f, -0.5f),
		vec3(0.5f, -0.5f,  0.5f)),
		Triangle(vec3(0.5f, -0.5f,  0.5f),
		vec3(-0.5f, -0.5f,  0.5f),
		vec3(-0.5f, -0.5f, -0.5f)),

		Triangle(vec3(-0.5f,  0.5f, -0.5f),
		vec3(0.5f,  0.5f, -0.5f),
		vec3(0.5f,  0.5f,  0.5f)),
		Triangle(vec3(0.5f,  0.5f,  0.5f),
		vec3(-0.5f,  0.5f,  0.5f),
		vec3(-0.5f,  0.5f, -0.5f))
	};

	vec3 lightPos = vec3(0, 0, 1);

	std::vector<std::thread*> pool;
	int num_threads = 4;
	std::vector<vec3> thread_colors;
	for (int i = 0; i < num_threads; i++) {
		float r = randomf(0.5, 1);
		float g = randomf(0.5, 1);
		float b = randomf(0.5, 1);
		vec3 col(r, g, b);
		thread_colors.push_back(col);
	}

	FrameBuffer frameBuffer(wsize.m_x, wsize.m_y);

	float angle = 0;

	int frameCount = 0;
	float k = 0;
	float delta;
	Timer timer;
	while (true) {
		frameCount++;

		delta = timer.elapsed();

		if (frameCount == 100) {
			std::string str;
			std::stringstream ss;
			ss << "FPS: " << 1 / delta << " Frametime: " << delta * 1000 << " Threads: " << pool.size();
			str = ss.str();
			window.SetTitle(str);
			frameCount = 0;
		}

		k += 0 * delta;
		lightPos.m_x = sin(k) * 2;

		if (pool.empty() == false) {
			for (auto& item : pool) {
				delete item;
			}
			pool.clear();
		}

		angle += 10 * PI / 180.0f * delta;

		std::vector<float> trans(16, 0.0);
		Identity(trans);
		
		trans = Translate(trans, vec3(0, 0, -1));
		trans = RotateX(trans, angle);
		
		Mesh subcube;
		subcube.triangles = cube.triangles;
		for (auto& triangle: subcube.triangles) {
			for (auto& vertex : triangle.vertices) {
				vec4 v = vec4(vertex, 1.0f);
				vertex = mul4x4(trans, v);
			}

			triangle.calcNormal();
		}

		frameBuffer.Clear();
		
		for (size_t x = 0; x < wsize.m_x; x++) {
			for (size_t y = 0; y < wsize.m_y; y++) {
				float rx = (x * 2) / wsize.m_x - 1;
				float ry = (y * 2) / wsize.m_y - 1;
				rx *= aspect;

				Ray rr;
				rr.origin = vec3(0, 0, 1);
				rr.dir = vec3(rx, ry, 0) - rr.origin;
				rr.dir.Normalize();

				for (auto& triangle : subcube.triangles) {
					float t = 0;
					if (intersectTriangleRay(rr, triangle, t)) {
						vec3 FragPos = rr.dir * t;
						vec3 lightDir = lightPos - FragPos;
						lightDir.Normalize();
						float diff = max(lightDir.Dot(triangle.normal), 0);
						if (diff > 0) color = vec3(0, 1, 0);
						else color = vec3(1, 0, 0);
						vec3 diffuse = vec3(1.0f, 1.0f, 1.0f) * diff;
						vec3 result = vec3(diffuse.m_x * color.m_x, diffuse.m_y * color.m_y, diffuse.m_z * color.m_z);

						frameBuffer.SetPixel(x, y, result);
					}
				}
			}
		}

		/*for (int i = 0; i < num_threads; i++) {
			color = thread_colors[i];
			std::thread* t = new std::thread(
				[i, &wsize, &num_threads, &aspect, &cube, &render_cube, &lightPos, &color, &frameBuffer]() {
					int chunkSize = wsize.m_y / num_threads;
					for (size_t x = 0; x < wsize.m_x; x++) {
						for (size_t y = i * chunkSize; y < i * chunkSize + chunkSize; y++) {
							float rx = (x * 2) / wsize.m_x - 1;
							float ry = (y * 2) / wsize.m_y - 1;
							rx *= aspect;

							Ray rr;
							rr.origin = vec3(0, 0, 1);
							rr.dir = vec3(rx, ry, 0) - rr.origin;
							rr.dir.Normalize();

							for (unsigned int vertex_index = 0, normal_index = 0; vertex_index < cube.vertices.size(); 
								vertex_index +=3, normal_index++) {
								if (intersectTriangleRay(rr, render_cube.vertices[vertex_index], render_cube.vertices[vertex_index+1], render_cube.vertices[vertex_index+2])) {
									vec3 FragPos = vec3(rx, ry, 0);
									vec3 lightDir = lightPos - FragPos;
									lightDir.Normalize();
									float diff = max(lightDir.Dot(render_cube.normals[normal_index]), 0);
									if (diff > 0) color = vec3(0, 1, 0);
									else color = vec3(1, 0, 0);
									vec3 diffuse = vec3(1.0f, 1.0f, 1.0f) * diff;
									vec3 result = vec3(diffuse.m_x * color.m_x, diffuse.m_y * color.m_y, diffuse.m_z * color.m_z);

									frameBuffer.SetPixel(x, y, result);
								}
							}
						}
					}

				}
			);

			pool.push_back(t);
		}

		for (int i = 0; i < num_threads; i++) {
			pool[i]->join();
		}*/

		window.Render(frameBuffer);
	}

	return 0;
}