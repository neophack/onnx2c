#include "aixlog.hpp"
#include <cassert>

#ifdef ERROR
#undef ERROR
#endif

#define ERROR(why)                       \
	do {                                 \
		LOG(FATAL) << why << std::flush; \
		assert(false);                   \
		exit(1);                         \
	} while (0)
