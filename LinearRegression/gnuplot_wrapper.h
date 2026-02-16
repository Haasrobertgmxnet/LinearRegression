#pragma once
#include <span>

class GnuplotWrapper {
public:
	GnuplotWrapper() {
#ifdef _WIN32
		gp = _popen("gnuplot -persistent", "w");
#else
		gp = popen("gnuplot -persistent", "w");
#endif
	}
	~GnuplotWrapper() {
#ifdef _WIN32
		_pclose(gp);
#else
		pclose(gp);
#endif
	}

	bool is_open() const {
		return gp != nullptr;
	}

	void send_command(const char* cmd) {
		if (gp == nullptr) {
			fprintf(stderr, "Error: Gnuplot not initialized.\n");
			return;
		}
		fprintf(gp, "%s\n", cmd);
		fflush(gp);
	}
	void plot(const std::span<double>& x, const std::span<double>& y) {};

private:
	FILE* gp{};
};