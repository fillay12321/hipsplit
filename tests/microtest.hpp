#ifndef HIPSPLIT_MICROTEST_HPP
#define HIPSPLIT_MICROTEST_HPP

// A no-dependency test harness. Just enough to keep the module self-contained;
// swap it for GoogleTest if you'd rather, the tests don't care. Each TEST()
// registers itself at static-init time and run_all() walks the list.

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace microtest {

struct Case {
    std::string           name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

inline int& failures() {
    static int f = 0;
    return f;
}

struct Reg {
    Reg(const std::string& n, std::function<void()> f) { registry().push_back({n, f}); }
};

inline void fail(const char* file, int line, const std::string& msg) {
    std::printf("  FAIL %s:%d  %s\n", file, line, msg.c_str());
    ++failures();
}

inline int run_all() {
    int failed_cases = 0;
    for (auto& c : registry()) {
        const int before = failures();
        std::printf("[ run ] %s\n", c.name.c_str());
        c.fn();
        if (failures() == before) std::printf("[  ok ] %s\n", c.name.c_str());
        else                      ++failed_cases;
    }
    std::printf("\n%d case(s) failed\n", failed_cases);
    return failed_cases == 0 ? 0 : 1;
}

} // namespace microtest

#define TEST(name)                                                       \
    static void name();                                                  \
    static microtest::Reg reg_##name(#name, name);                       \
    static void name()

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond))                                                     \
            microtest::fail(__FILE__, __LINE__, "CHECK failed: " #cond); \
    } while (0)

#define CHECK_EQ(a, b)                                                   \
    do {                                                                 \
        auto _va = (a);                                                  \
        auto _vb = (b);                                                  \
        if (!(_va == _vb))                                               \
            microtest::fail(__FILE__, __LINE__,                          \
                            "CHECK_EQ failed: " #a " != " #b);           \
    } while (0)

#endif // HIPSPLIT_MICROTEST_HPP
