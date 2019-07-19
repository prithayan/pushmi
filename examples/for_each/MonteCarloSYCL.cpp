////////////////////////////////////////////////////////////////////////////////
//
//  Monte Carlo Sycl 
//
//  Author: Kate McGrievy, March 2018
//
////////////////////////////////////////////////////////////////////////////////

/*!
Implementation of a MonteCarlo Simulation for European style stock options.
Uses a Mersenne Twister and  Box-Muller to generate the random stock moves.
*/



//#include "CL/sycl.hpp"

#include <iostream>
#include <array>
#include <chrono> 
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <complex>
#include <string.h>
#include <string>
#include <exception>
#include <stdexcept>
#include <cmath>

#include "../for_each.h"
//#include <pushmi/o/for_each.h>
#include "../pool.h"

#ifndef M_PI
#define M_PI ((float)3.14159265358979)
#endif

#define NUM_OF_OPTIONS 16384
#define NUM_SAMP 8
#define NUM_OF_WORKGROUPS 16
#define NUM_OF_RUNS 5

#define HALF 0.5f
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f


// scaling factor 2^32
#define DIVISOR 4294967296.0f
#define UINT_THIRTY 30U
#define UINT_BIG_CONST 1812433253U
#define MN 624
#define M 397

// Mersenne twister constant 2567483615
#define MATRIX_A 0x9908B0DFU
// Mersenne twister constant 2636928640
#define MASK_B 0x9D2C5680U
// Mersenne twister constant 4022730752
#define MASK_C 0xEFC60000U
// Mersenne twister tempering shifts
#define SHIFT_U 11
#define SHIFT_S 7
#define SHIFT_T 15
#define SHIFT_L 18
#define CHECK_VALIDITY true
#define EPSILION 0.1f

#define VOLATILITY 0.2f
#define RISKFREERATE 0.05f
#define NSAMPLES 262144

//#include "basic.hpp" 


#include <iostream>
#include <exception>
#include <vector>
#include <cerrno>
#include <cstdint>
//#include <CL/cl_gl.h>

#include "basic.hpp"


#ifdef __linux__
#include <sys/time.h>
#include <unistd.h>
#include <libgen.h>
#elif defined(_WIN32) || defined(WIN32)
#include <Windows.h>
#else
#include <ctime>
#endif

using std::string;

using namespace pushmi::aliases;
using namespace std::chrono; 


auto inline_bulk_target() {
  return [](auto init,
            auto selector,
            auto input,
            auto&& func,
            auto sb,
            auto se,
            auto out) {
    try {
      auto acc = init(input);
      unsigned index = 0 ; 
      for (decltype(sb) idx{sb}; idx != se; ++idx, ++index) {
        func(acc, idx, index);
      }
      auto result = selector(std::move(acc));
      mi::set_value(out, std::move(result));
      mi::set_done(out);
    } catch (...) {
      mi::set_error(out, std::current_exception());
    }
  };
}

template <class Executor, class Allocator = std::allocator<char>>
auto naive_executor_bulk_target(Executor e, Allocator a = Allocator{}) {
  return [e, a](
             auto init,
             auto selector,
             auto input,
             auto&& func,
             auto sb,
             auto se,
             auto out) mutable {
    using RS = decltype(selector);
    using F = std::conditional_t<
        std::is_lvalue_reference<decltype(func)>::value,
        decltype(func),
        typename std::remove_reference<decltype(func)>::type>;
    using Out = decltype(out);
    try {
      typename std::allocator_traits<Allocator>::template rebind_alloc<char>
          allocState(a);
      auto shared_state = std::allocate_shared<std::tuple<
          std::exception_ptr, // first exception
          Out, // destination
          RS, // selector
          F, // func
          std::atomic<decltype(init(input))>, // accumulation
          std::atomic<std::size_t>, // pending
          std::atomic<std::size_t> // exception count (protects assignment to
                                   // first exception)
          >>(
          allocState,
          std::exception_ptr{},
          std::move(out),
          std::move(selector),
          (decltype(func)&&)func,
          init(std::move(input)),
          1,
          0);
      e.schedule() | op::submit([e, sb, se, shared_state](auto) mutable {
        auto stepDone = [](auto shared_state) {
          // pending
          if (--std::get<5>(*shared_state) == 0) {
            // first exception
            if (std::get<0>(*shared_state)) {
              mi::set_error(
                  std::get<1>(*shared_state), std::get<0>(*shared_state));
              return;
            }
            try {
              // selector(accumulation)
              auto result = std::get<2>(*shared_state)(
                  std::move(std::get<4>(*shared_state).load()));
              mi::set_value(std::get<1>(*shared_state), std::move(result));
              mi::set_done(std::get<1>(*shared_state));
            } catch (...) {
              mi::set_error(
                  std::get<1>(*shared_state), std::current_exception());
            }
          }
        };
         unsigned index=0;
        for (decltype(sb) idx{sb}; idx != se; ++idx, index++) {
          ++std::get<5>(*shared_state);
          e.schedule() | op::submit([shared_state, idx, stepDone, index](auto ex) {
            try {
              // this indicates to me that bulk is not the right abstraction
              auto old = std::get<4>(*shared_state).load();
              auto step = old;
              do {
                step = old;
                // func(accumulation, idx)
                //std::cout<<"\n line 95:";
                std::get<3> (*shared_state)(step, idx, index);
              } while (!std::get<4>(*shared_state)
                            .compare_exchange_strong(old, step));
            } catch (...) {
              // exception count
              if (std::get<6>(*shared_state)++ == 0) {
                // store first exception
                std::get<0>(*shared_state) = std::current_exception();
              } // else eat the exception
            }
            stepDone(shared_state);
          });
        }
        stepDone(shared_state);
      });
    } catch (...) {
      e.schedule() |
          op::submit([out = std::move(out), ep = std::current_exception()](
                         auto) mutable { mi::set_error(out, ep); });
    }
  };
}
void* aligned_malloc (size_t size, size_t alignment)
{
    // a number of requirements should be met
    assert(alignment > 0);
    assert((alignment & (alignment - 1)) == 0); // test for power of 2

    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    assert(size >= sizeof(void*));
    assert(size/sizeof(void*)*sizeof(void*) == size);

    // allocate extra memory and convert to size_t to perform calculations
    char* orig = new char[size + alignment + sizeof(void*)];
    // calculate an aligned position in the allocated region
    // assumption: (size_t)orig does not lose lower bits
    char* aligned =
        orig + (
        (((size_t)orig + alignment + sizeof(void*)) & ~(alignment - 1)) -
        (size_t)orig
        );
    // save the original pointer to use it in aligned_free
    *((char**)aligned - 1) = orig;
    return aligned;
}


void aligned_free (void *aligned)
{
    if(!aligned)return; // behaves as delete: calling with 0 is NOP
    delete [] *((char**)aligned - 1);
}


bool is_number (const string& x)
{
    // Detection is simple: just try to represent x as an int
    try
    {
        // If x is a number, then str_to returns without an exception
        // In case when x cannot be converted to int
        // str_to rises Error exception (see str_to definitin)
        str_to<int>(x);

        // success: x is a number
        return true;
    }
    catch(const Error&)
    {
        // fail: x is not a number
        return false;
    }
}


double time_stamp ()
{
#ifdef __linux__
    {
        struct timeval t;
        if(gettimeofday(&t, 0) != 0)
        {
            throw Error(
                "Linux-specific time measurement counter (gettimeofday) "
                "is not available."
                );
        }
        return t.tv_sec + t.tv_usec/1e6;
    }
#elif defined(_WIN32) || defined(WIN32)
    {
        LARGE_INTEGER curclock;
        LARGE_INTEGER freq;
        if(
            !QueryPerformanceCounter(&curclock) ||
            !QueryPerformanceFrequency(&freq)
            )
        {
            throw Error(
                "Windows-specific time measurement counter (QueryPerformanceCounter, "
                "QueryPerformanceFrequency) is not available."
                );
        }

        return double(curclock.QuadPart)/freq.QuadPart;
    }
#else
    {
        // very low resolution
        return double(time(0));
    }
#endif
}


void destructorException ()
{
    if(std::uncaught_exception())
    {
        // don't crash an application because of double throwing
        // let the user see the original exception and suppress
        // this one instead
        std::cerr
            << "[ ERROR ] Catastrophic: another exception "
            << "was thrown and suppressed during handling of "
            << "previously started exception raising process.\n";
    }
    else
    {
        // that's OK, go up!
        throw;
    }
}

/*
cl_uint zeroCopyPtrAlignment (cl_device_id device)
{
    // Please refer to Intel Zero Copy Tutorial and OpenCL Performance Guide
    return 4096;
}
*/

/*
size_t zeroCopySizeAlignment (size_t requiredSize, cl_device_id device)
{
    // Please refer to Intel Zero Copy Tutorial and OpenCL Performance Guide
    // The following statement rounds requiredSize up to the next 64-byte boundary
    return requiredSize + (~requiredSize + 1) % 64;   // or even shorter: requiredSize + (-requiredSize) % 64
}
*/

bool verifyZeroCopyPtr (void* ptr, size_t sizeOfContentsOfPtr)
{
    return                                  // To enable zero-copy for buffer objects
        (std::uintptr_t)ptr % 4096  ==  0   // pointer should be aligned to 4096 bytes boundary
        &&                                  // and
        sizeOfContentsOfPtr % 64  ==  0;    // size of memory should be aligned to 64 bytes boundary.
}

/*
cl_uint requiredOpenCLAlignment (cl_device_id device)
{
    cl_uint result = 0;
    cl_int err = clGetDeviceInfo(
        device,
        CL_DEVICE_MEM_BASE_ADDR_ALIGN,
        sizeof(result),
        &result,
        0
        );
    SAMPLE_CHECK_ERRORS(err);
    assert(result%8 == 0);
    return result/8;    // clGetDeviceInfo returns value in bits, convert it to bytes
}
*/

/*
size_t deviceMaxWorkGroupSize (cl_device_id device)
{
    size_t result = 0;
    cl_int err = clGetDeviceInfo(
        device,
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(result),
        &result,
        0
        );
    SAMPLE_CHECK_ERRORS(err);
    return result;
}


void deviceMaxWorkItemSizes (cl_device_id device, size_t* sizes)
{
    cl_int err = clGetDeviceInfo(
        device,
        CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizeof(size_t[3]),
        sizes,
        0
        );
    SAMPLE_CHECK_ERRORS(err);
}


size_t kernelMaxWorkGroupSize (cl_kernel kernel, cl_device_id device)
{
    size_t result = 0;
    cl_int err = clGetKernelWorkGroupInfo(
        kernel,
        device,
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(result),
        &result,
        0
        );
    SAMPLE_CHECK_ERRORS(err);
    return result;
}


double eventExecutionTime (cl_event event)
{
    cl_ulong end = 0, start = 0;

    cl_int err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, 0);
    SAMPLE_CHECK_ERRORS(err);

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, 0);
    SAMPLE_CHECK_ERRORS(err);

    return double(end - start)/1e9; // convert in seconds
}
*/

string exe_dir ()
{
    using namespace std;
    const int start_size = 1000;
    const int max_try_count = 8;

#ifdef __linux__
    {
        static string const exe = "/proc/self/exe";

        vector<char> path(start_size);
        int          count = max_try_count;  // Max number of iterations.

        for(;;)
        {
            ssize_t len = readlink(exe.c_str(), &path[0], path.size());

            if(len < 0)
            {
                throw Error(
                    "Cannot retrieve path to the executable: "
                    "readlink returned error code " +
                    to_str(errno) + "."
                    );
            }

            if(len < path.size())
            {
                // We got the path.
                path.resize(len);
                break;
            }

            if(count > 0)   // the buffer is too small
            {
                --count;
                // Enlarge the buffer.
                path.resize(path.size() * 2);
            }
            else
            {
                throw Error("Cannot retrieve path to the executable: path is too long.");
            }
        }

        return string(dirname(&path[0])) + "/";
    }
#elif defined(_WIN32) || defined(WIN32)
    {
        // Retrieving path to the executable.

        vector<char> path(start_size);
        int count = max_try_count;

        for(;;)
        {
            DWORD len = GetModuleFileNameA(NULL, &path[0], (DWORD)path.size());

            if(len == 0)
            {
                int err = GetLastError();
                throw Error(
                    "Getting executable path failed with error " +
                    to_str(err)
                    );
            }

            if(len < path.size())
            {
                path.resize(len);
                break;
            }

            if(count > 0)   // buffer is too small
            {
                --count;
                // Enlarge the buffer.
                path.resize(path.size() * 2);
            }
            else
            {
                throw Error(
                    "Cannot retrieve path to the executable: "
                    "path is too long."
                    );
            }
        }

        string exe(&path[0], path.size());

        // Splitting the path into components.

        vector<char> drv(_MAX_DRIVE);
        vector<char> dir(_MAX_DIR);
        count = max_try_count;

        for(;;)
        {
            int rc =
                _splitpath_s(
                exe.c_str(),
                &drv[0], drv.size(),
                &dir[0], dir.size(),
                NULL, 0,   // We need neither name
                NULL, 0    // nor extension
                );
            if(rc == 0)
            {
                break;
            }
            else if(rc == ERANGE)
            {
                if(count > 0)
                {
                    --count;
                    // Buffer is too small, but it is not clear which one.
                    // So we have to enlarge both.
                    drv.resize(drv.size() * 2);
                    dir.resize(dir.size() * 2);
                }
                else
                {
                    throw Error(
                        "Getting executable path failed: "
                        "Splitting path " + exe + " to components failed: "
                        "Buffers of " + to_str(drv.size()) + " and " +
                        to_str(dir.size()) + " bytes are still too small."
                        );
                }
            }
            else
            {
                throw Error(
                    "Getting executable path failed: "
                    "Splitting path " + exe +
                    " to components failed with code " + to_str(rc)
                    );
            }
        }

        // Combining components back to path.
        return string(&drv[0]) + string(&dir[0]);
    }
#else
    {
        throw Error(
            "There is no method to retrieve the directory path "
            "where executable is placed: unsupported platform."
            );
    }
#endif
}

std::wstring exe_dir_w ()
{
    using namespace std;
    const int start_size = 1000;
    const int max_try_count = 8;

#if defined(_WIN32) || defined(WIN32)
    {
        // Retrieving path to the executable.

        vector<wchar_t> path(start_size);
        int count = max_try_count;

        for(;;)
        {
            DWORD len = GetModuleFileNameW(NULL, &path[0], (DWORD)path.size());

            if(len == 0)
            {
                int err = GetLastError();
                throw Error(
                    "Getting executable path failed with error " +
                    to_str(err)
                    );
            }

            if(len < path.size())
            {
                path.resize(len);
                break;
            }

            if(count > 0)   // buffer is too small
            {
                --count;
                // Enlarge the buffer.
                path.resize(path.size() * 2);
            }
            else
            {
                throw Error(
                    "Cannot retrieve path to the executable: "
                    "path is too long."
                    );
            }
        }

        wstring exe(&path[0], path.size());

        // Splitting the path into components.

        vector<wchar_t> drv(_MAX_DRIVE);
        vector<wchar_t> dir(_MAX_DIR);
        count = max_try_count;

        for(;;)
        {
            int rc =
                _wsplitpath_s(
                exe.c_str(),
                &drv[0], drv.size(),
                &dir[0], dir.size(),
                NULL, 0,   // We need neither name
                NULL, 0    // nor extension
                );
            if(rc == 0)
            {
                break;
            }
            else if(rc == ERANGE)
            {
                if(count > 0)
                {
                    --count;
                    // Buffer is too small, but it is not clear which one.
                    // So we have to enlarge both.
                    drv.resize(drv.size() * 2);
                    dir.resize(dir.size() * 2);
                }
                else
                {
                    throw Error(
                        "Getting executable path failed: "
                        "Splitting path " + wstringToString(exe) + " to components failed: "
                        "Buffers of " + to_str(drv.size()) + " and " +
                        to_str(dir.size()) + " bytes are still too small."
                        );
                }
            }
            else
            {
                throw Error(
                    "Getting executable path failed: "
                    "Splitting path " + wstringToString(exe) +
                    " to components failed with code " + to_str(rc)
                    );
            }
        }

        // Combining components back to path.
        return wstring(&drv[0]) + wstring(&dir[0]);
    }
#else
    {
        throw Error(
            "There is no method to retrieve the directory path "
            "where executable is placed: unsupported platform."
            );
    }
#endif
}

std::wstring stringToWstring (const std::string s)
{
    return std::wstring(s.begin(), s.end());
}

#ifdef __linux__
std::string wstringToString (const std::wstring w)
{
    string tmp;
    const wchar_t* src = w.c_str();
    //Store current locale and set default locale
    CTYPELocaleHelper locale_helper;

    //Get required number of characters
    size_t count = wcsrtombs(NULL, &src, 0, NULL);
    if(count == size_t(-1))
    {
        throw Error(
            "Cannot convert wstring to string"
        );
    }
    std::vector<char> dst(count+1);

    //Convert wstring to multibyte representation
    size_t count_converted = wcsrtombs(&dst[0], &src, count+1, NULL);
    dst[count_converted] = '\0';
    tmp.append(&dst[0]);
    return tmp;  
}
#else
std::string wstringToString (const std::wstring w)
{
    string tmp;

    const char* question_mark = "?"; //replace wide characters which don't fit in the string with "?" mark

    for(unsigned int i = 0; i < w.length(); i++)
    {
        if(w[i]>255||w[i]<0)
        {
            tmp.append(question_mark);
        }
        else
        {
            tmp += (char)w[i];
        }
    }
    return tmp;  
}
#endif

size_t round_up_aligned (size_t x, size_t alignment)
{
    assert(alignment > 0);
    assert((alignment & (alignment - 1)) == 0); // test for power of 2

    size_t result = (x + alignment - 1) & ~(alignment - 1);

    assert(result >= x);
    assert(result - x < alignment);
    assert((result & (alignment - 1)) == 0);

    return result;
}
//using namespace cl::sycl;
using namespace std;



namespace
{
	/* Based on C++ implementation from
	http://www.johndcook.com/cpp_phi.html
	*/
	
	float cdf(float x)
	{
		// constants
		float a1 = 0.254829592f;
		float a2 = -0.284496736f;
		float a3 = 1.421413741f;
		float a4 = -1.453152027f;
		float a5 = 1.061405429f;
		float p = 0.3275911f;
		// Save the sign of x
		int sign = 1;
		if (x < 0)
			sign = -1;
		x = abs(x) / std::sqrt(2.0f);

		// A&S formula 7.1.26
		// A&S refers to Handbook of Mathematical Functions by Abramowitz and Stegun
		float t = 1.0f / (1.0f + p*x);
		float y = 1.0f - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t* std::exp(-x*x);

		return 0.5f*(1.0f + sign*y);
	}
}


// Reference Black Scholes formula implementation (vcall_ref, vput_ref) and comparison with option prices calculated
// using Monte Carlo based options pricing method implementation in OpenCL (vcall, vput)

bool checkValidity(float nopt,
	float r, float sig, float *s0, float *x,
	float *t, float *vcall, float *vput, float *vcall_ref, float *vput_ref, float threshold)
{
	float a, b, c, y, z, e, d1, d2, w1, w2;
	using namespace std;
	//const T HALF = 0.5f;

	for (size_t i = 0; i < nopt; i++)
	{
		a = std::log(s0[i] / x[i]);
		b = t[i] * r;
		z = t[i] * sig*sig;

		c = HALF * z;
		e = std::exp(-b);
		y = 1.0f / std::sqrt(z);

		w1 = (a + b + c) * y;
		w2 = (a + b - c) * y;

		d1 = 2.0f*cdf(w1) - 1.0f;
		d2 = 2.0f*cdf(w2) - 1.0f;

		d1 = HALF + HALF*d1;
		d2 = HALF + HALF*d2;

		vcall_ref[i] = s0[i] * d1 - x[i] * e*d2;
		vput_ref[i] = vcall_ref[i] - s0[i] + x[i] * e;
		
		if (abs(vcall_ref[i] - vcall[i]) > threshold)
		{
			cerr << "VALIDAION FAILED!!!\n vcall_ref" << "[" << i << "] = "
				<< vcall_ref[i] << ", vcall" << "[" << i << "] = "
				<< vcall[i] << "\n"
				<< "Probably validation threshold = " << threshold << " is too small\n";
			return false;
		}
		if (abs(vput_ref[i] - vput[i]) > threshold)
		{
			cerr << "VALIDAION FAILED!!!\n vput_ref" << "[" << i << "] = "
				<< vput_ref[i] << ", vput" << "[" << i << "] = "
				<< vput[i] << "\n"
				<< "Probably validation threshold = " << threshold << " is too small\n";
			return false;
		}
		
	}

	std::cout << "VALIDATION PASSED\n";
	return true;
}


//cl::sycl::device selectedDevice(int plateform_id, int device_id){
//
//	std::vector<cl::sycl::platform> platforms = cl::sycl::platform::get_platforms();
//	cl::sycl::device selected_device;	
//
//    std::cout << "Platform:" << std::endl;
//    std::cout << "+--Device:" << std::endl;
//    std::cout << "---------------------------------" << std::endl;
//    
//    bool device_found = false;
//
//    for (int plateform_index = 0; plateform_index < platforms.size(); plateform_index++){
//        
//        std::string name = platforms[plateform_index].get_info<info::platform::name>();
//        std::vector<cl::sycl::device> devices = platforms[plateform_index].get_devices(info::device_type::all);   
//        bool is_selected_plateform = 0;
//        if (plateform_id != plateform_index){
//	   		std::cout << "[" << plateform_index << "]" << name << std::endl;
//            
//		} else {
//			std::cout << "[X]" << name << std::endl;
//		}
//    
//        for (int device_index = 0; device_index < devices.size(); device_index++){
//            cl::sycl::device device = devices[device_index];
//            std::string device_name = device.get_info<cl::sycl::info::device::name>(); 
//            
//            if (!((device_id == device_index) && (plateform_id == plateform_index))) { 
//                std::cout << "+--[" << device_index  << "]" << device_name << std::endl;
//            } else {
//                std::cout << "+--[X]" << device_name << std::endl;
//                selected_device = device; 
//                device_found = true;
//            }         
//        }
//    }
//
//	return selected_device; 
//}   


void computeOptionPrices(unsigned wiID, float riskFreeRate,  
	float sigma, float* time_buf,
	float* sPrice_buf, float* kPrice_buf,
	float* callPrice_buf, float* putPrice_buf)
{
	
	float lastRand = ZERO;
	float currentCallPrice = ZERO;
	float currentPutPrice = ZERO;

	float t = time_buf[wiID];
	float stockPrice = sPrice_buf[wiID];
	float kprice = kPrice_buf[wiID];

	float int_to_float_normalize_factor = (ONE / DIVISOR);

	float tmp_bs1 = (riskFreeRate - sigma*sigma*HALF)*t;
	float tmp_bs2 = sigma* std::sqrt(t);

	

	// State indexes
	int i, iCurrentMiddle, iCurrent;
	// Mersenne twister generated random number
	uint mt_rnd_num;
	// State of the MT generator
	int mt_state[MN];
	// Temporary state for MT states swap
	int tmp_mt;

	// set seed
	mt_state[0] = (int)wiID;
	iCurrent = 0;

	// Initialize the MT generator from a seed
	for (i = 1; i < MN; i++)
	{
		mt_state[i] = (int)i + UINT_BIG_CONST * (mt_state[i - 1] ^ (mt_state[i - 1] >> UINT_THIRTY));
	}

	i = 0;
	tmp_mt = mt_state[0];

	//callPrice_buf[item.get_global(0)] = 1.0f;
		//putPrice_buf[item.get_global(0)] = 1.0f;

	for (int iSample = 0; iSample < NSAMPLES; iSample += 2) // Generate two samples per iteration as it is convinient for Box-Muller
	{
		iCurrent = (iCurrent == MN - 1) ? 0 : i + 1;
		iCurrentMiddle = (i + M >= MN) ? i + M - MN : i + M;

		mt_state[i] = tmp_mt;
		tmp_mt = mt_state[iCurrent];

		// MT recurrence
		// Generate untempered numbers
		mt_rnd_num = (mt_state[i] & 0x80000000U) | (mt_state[iCurrent] & 0x7FFFFFFFU);
		mt_rnd_num = mt_state[iCurrentMiddle] ^ (mt_rnd_num >> 1) ^ ((0 - (mt_rnd_num & 1)) & MATRIX_A);
		mt_state[i] = mt_rnd_num;

		// Tempering pseudorandom number
		mt_rnd_num ^= (mt_rnd_num >> SHIFT_U);
		mt_rnd_num ^= (mt_rnd_num << SHIFT_S) & MASK_B;
		mt_rnd_num ^= (mt_rnd_num << SHIFT_T) & MASK_C;
		mt_rnd_num ^= (mt_rnd_num >> SHIFT_L);

		float rnd_num = (float)mt_rnd_num;
		i = iCurrent;

		// Second MT random number generation
		// Calculate new state indexes
		iCurrent = (iCurrent == MN - 1) ? 0 : i + 1;
		iCurrentMiddle = (i + M >= MN) ? i + M - MN : i + M;

		mt_state[i] = tmp_mt;
		tmp_mt = mt_state[iCurrent];

		// MT recurrence
		// Generate untempered numbers
		mt_rnd_num = (mt_state[i] & 0x80000000U) | (mt_state[iCurrent] & 0x7FFFFFFFU);
		mt_rnd_num = mt_state[iCurrentMiddle] ^ (mt_rnd_num >> 1) ^ ((0 - (mt_rnd_num & 1)) & MATRIX_A);

		mt_state[i] = mt_rnd_num;

		// Tempering pseudorandom number
		mt_rnd_num ^= (mt_rnd_num >> SHIFT_U);
		mt_rnd_num ^= (mt_rnd_num << SHIFT_S) & MASK_B;
		mt_rnd_num ^= (mt_rnd_num << SHIFT_T) & MASK_C;
		mt_rnd_num ^= (mt_rnd_num >> SHIFT_L);

		float rnd_num1 = (float)mt_rnd_num;

		i = iCurrent;

		rnd_num = (rnd_num + ONE) * int_to_float_normalize_factor;
		rnd_num1 = (rnd_num1 + ONE) * int_to_float_normalize_factor;

		float tmp_bm = std::sqrt(std::fmax(-TWO*std::log(rnd_num), ZERO)); // max added to be sure that sqrt argument non-negative

		rnd_num = tmp_bm*std::cos(TWO*((float)M_PI)*rnd_num1);
		rnd_num1 = tmp_bm*std::sin(TWO*((float)M_PI)*rnd_num1);

		

		// Stock price formula
		// Add first sample from pair
		float tmp_bs3 = rnd_num*tmp_bs2 + tmp_bs1; // formula reference: NormalDistribution*sigma*(T)^(1/2) + (r - (sigma^2)/2)*(T)
		tmp_bs3 = stockPrice * std::exp(tmp_bs3); // formula reference: S * exp(CND*sigma*(T)^(1/2) + (r - (sigma^2)/2)*(T))


		float dif_call = tmp_bs3 - kprice;
		currentCallPrice += std::fmax(dif_call, ZERO);


		// Add second sample from pair
		tmp_bs3 = rnd_num1*tmp_bs2 + tmp_bs1;
		tmp_bs3 = stockPrice * std::exp(tmp_bs3);
		dif_call = tmp_bs3 - kprice;

		currentCallPrice += std::fmax(dif_call, ZERO);

	} 
	
	
	currentCallPrice = currentCallPrice / ((float)NSAMPLES) * std::exp(-riskFreeRate*t);
	currentPutPrice = currentCallPrice - stockPrice + kprice * std::exp(-riskFreeRate*t); 

	callPrice_buf[0] = currentCallPrice;
	putPrice_buf[0] = currentPutPrice;
	
}


void mc(int deviceID, int plateformID) {

	 
  unsigned MAXTHREADS = std::thread::hardware_concurrency();
  std::ofstream csvFile; 
  csvFile.open("executorRuntime.csv", std::ios::out|std::ios::app);
  csvFile<<std::fixed;

	const size_t noptions = (int)NUM_OF_OPTIONS;

	int nsamples = NSAMPLES;
	std::cout << "Running Monte Carlo options pricing for " << noptions << " options, with " << nsamples << " samples\n";
	std::cout << "Total number of runs " << NUM_OF_RUNS << std::endl;

	
	//std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
	//std::cout << "Devices:" << std::endl;
	  
	//cl::sycl::device selected_device = selectedDevice(plateformID, deviceID);	
	
	std::srand (std::time(NULL));
	
	float *time;
	time = new float[noptions]; 

	float *callPrice;
	callPrice = new float[noptions]; 

	float *putPrice;
	putPrice =  new float[noptions];  

	float *ref_callPrice;
	ref_callPrice = new float[noptions]; 

	float *ref_putPrice;
	ref_putPrice = new float[noptions];

	float *sPrice;
	sPrice = new float[noptions];

	float *kPrice;
	kPrice = new float[noptions]; 

	fill_rand_uniform_01(sPrice, noptions);
	float S0L = 10.0f;
	float S0H = 50.0f;
	for (size_t i = 0; i < noptions; i++)
	{
		sPrice[i] = sPrice[i] * (S0H - S0L) + S0L;
	}
	
	fill_rand_uniform_01(kPrice, noptions);
	float XL = 10.0f;
	float XH = 50.0f;
	for (size_t i = 0; i < noptions; i++)
	{
		kPrice[i] = kPrice[i] * (XH - XL) + XL;
	}
	
	fill_rand_uniform_01(time, noptions);
	float TL = 0.2f;
	float TH = 2.0f;
	for (size_t i = 0; i < noptions; i++)
	{
		time[i] = time[i] * (TH - TL) + TL;
	}
	
	for (size_t i = 0; i < noptions; i++)
	{
		putPrice[i] = callPrice[i] = ref_callPrice[i] = ref_putPrice[i] = 0;
	}
	
	auto start = std::chrono::high_resolution_clock::now();

	//const cl::sycl::context new_context = cl::sycl::context(selected_device, NULL);
	//auto propList = cl::sycl::property_list{property::queue::enable_profiling()};
	//queue deviceQueue(selected_device, propList);

	//auto device = deviceQueue.get_device();
	//auto maxBlockSize = device.get_info<cl::sycl::info::device::max_work_group_size>();
	//auto maxComputeUnits = device.get_info<cl::sycl::info::device::max_compute_units>();
		
		
	//std::cout << "DEBUG: The Device Max Work Group Size is : " << maxBlockSize << std::endl;
	//std::cout << "DEBUG: The Device Max Compute Units is : " <<  maxComputeUnits << std::endl; 
	
	//int localsize =  (noptions / maxComputeUnits);
	//std::cout << "DEBUG: Local size is : " <<  localsize << std::endl;
	
	
	{
		
	float riskFreeRate = 0.05f;
	float volitility = 0.2f;
  auto numOfItems = noptions;
	//cl::sycl::range<1> numOfItems{ noptions };
	//cl::sycl::buffer<cl::sycl::cl_float, 1> bufferTime(time, numOfItems);
	//cl::sycl::buffer<cl::sycl::cl_float, 1> bufferCallPrice(callPrice, numOfItems);
	//cl::sycl::buffer<cl::sycl::cl_float, 1> bufferPutPrice(putPrice, numOfItems);
	//cl::sycl::buffer<cl::sycl::cl_float, 1> bufferSPrice(sPrice, numOfItems);
	//cl::sycl::buffer<cl::sycl::cl_float, 1> bufferKPrice(kPrice, numOfItems);

	
	
	try {	

			
			double elapsed = 0;
			
  for (unsigned MAXTHREADS = 2; MAXTHREADS <= std::thread::hardware_concurrency()+3; MAXTHREADS++){
        csvFile << "\n"<< MAXTHREADS <<",";
			for (int run = 0; run < (int)NUM_OF_RUNS; run++) {
			
				

				//cl::sycl::event queue_event = deviceQueue.submit([&](handler &cgh) 
        {

					//auto accessorTime = bufferTime.get_access<access::mode::read>(cgh);
					//auto accessorSPrice = bufferSPrice.get_access<access::mode::read>(cgh);
					//auto accessorKPrice = bufferKPrice.get_access<access::mode::read>(cgh);
					//auto accessorCallPrice = bufferCallPrice.get_access<access::mode::read_write>(cgh);
					//auto accessorPutPrice = bufferPutPrice.get_access<access::mode::read_write>(cgh);
					auto accessorTime = time ;
					auto accessorSPrice = sPrice ;
					auto accessorKPrice = kPrice;
					//auto accessorCallPrice = callPrice;
					//auto accessorPutPrice = putPrice;

          std::vector<std::vector<float>> Price_buf(numOfItems);
          for (unsigned i = 0 ; i < numOfItems ; i++){
            Price_buf[i].resize(2);
            Price_buf[i][0] = 0;
            Price_buf[i][1] = 0;
          }
          mi::pool p{std::max(1u, MAXTHREADS )};
        auto start = high_resolution_clock::now(); 
          mi::for_each(
            naive_executor_bulk_target(p.executor()),
            Price_buf.begin(),
            Price_buf.end(),
            [&](std::vector<float> &x, unsigned index)
            {
            float y; 
              float* accessorCallPrice= &x[0] ; 
              float* accessorPutPrice= &x[1];
              //std::cout<<"size of x::"<<x.size();
              //std::cout<<" "<<x[0] <<" and "<< x[1];
              computeOptionPrices(index, riskFreeRate, volitility, accessorTime, accessorSPrice,
                accessorKPrice, accessorCallPrice, accessorPutPrice);

            });
					//cgh.parallel_for<class monteCarloSimulation>(numOfItems,[=](cl::sycl::id<1> wiID)
          //for (unsigned int wiId = 0 ; wiId < numOfItems ; wiId++)
					//{
					//	
					//	computeOptionPrices(wiId, riskFreeRate, volitility, accessorTime, accessorSPrice,
					//		accessorKPrice, accessorCallPrice, accessorPutPrice);
					//}
          //);

						
        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start); 
	
        auto microTime = duration.count()/1000000.00;
			std::cout << "Total run time: " << microTime << " seconds" << std::endl; 
        csvFile << microTime <<",";
	        for (size_t i = 0; i < noptions; i++)
	        {
	        	 callPrice[i] = Price_buf[i][0]; 
              putPrice[i]=Price_buf[i][1];
	        }
				}
        //);
				
				//try {
				//	deviceQueue.wait_and_throw();
				//}
				//catch (cl::sycl::exception const& e) {
				//	std::cout << "Caught synchronous SYCL exception:\n" << e.what() << " " << std::endl;
				//}


				 
				
			//	cl::sycl::cl_ulong time_start, time_end;
			//	//Step 6.) Get the profiling information with the times.
			//	time_start = queue_event.template get_profiling_info<cl::sycl::info::event_profiling::command_start>();
      //      	time_end = queue_event.template get_profiling_info<cl::sycl::info::event_profiling::command_end>();
      //      	elapsed += (time_end - time_start)/1e9;
				 
			
			}
  }

			
			//cout << "Kernel Time: " << elapsed << " sec.\n";
			//cout << "Kernel perf: " << ((noptions * 5)/ elapsed) << " Options per second\n";
			//cout << "Average Kernel Time per run: " << (elapsed/NUM_OF_RUNS) << " sec.\n";
 

		}
	catch(std::exception e){
		std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
	}

	}

	//Do for only the last run.
	if (CHECK_VALIDITY == true) {
			
		checkValidity(noptions, RISKFREERATE, VOLATILITY, sPrice, kPrice, time, callPrice, putPrice, ref_callPrice, 
			ref_putPrice, EPSILION);	
	}
	
}


char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}


int main(int argc, char * argv[]) {
	
	try
	{
		int device_id = 0;
		int plateform_id = 0;

		if(cmdOptionExists(argv, argv+argc, "-d"))
    	{
        	std::string device_id_string = std::string(getCmdOption(argv, argv + argc, "-d"));
			std::string plateform_id_string = std::string(getCmdOption(argv, argv + argc, "-p"));  

			device_id = stoi(device_id_string);
			plateform_id = stoi(plateform_id_string); 

			std::cout << "Device ID being used: " + device_id << std::endl; 
		}

		mc(device_id, plateform_id);

		
  
  		return EXIT_SUCCESS;
	}
	catch (const std::exception& error)
	{
		cerr << "[ ERROR ] " << error.what() << "\n";
		return EXIT_FAILURE;
	}
	catch (...)
	{
		cerr << "[ ERROR ] Unknown/internal error happened.\n";
		return EXIT_FAILURE;
	}
}





