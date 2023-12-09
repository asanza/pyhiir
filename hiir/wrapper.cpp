#include "hiir/PolyphaseIir2Designer.h"

using namespace hiir;


#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

extern "C"
{
    DLL_EXPORT int iir2designer_compute_nbr_coefs_from_proto(double attenuation, double transition);
    DLL_EXPORT double iir2designer_compute_atten_from_order_tbw(int nbr_coefs, double transition);
    DLL_EXPORT int iir2designer_compute_coefs(double* coef_arr, double attenuation, double transition);
    DLL_EXPORT void iir2designer_compute_coefs_spec_order_tbw(double coef_arr[], int nbr_coefs, double transition);
}

int iir2designer_compute_nbr_coefs_from_proto(double attenuation, double transition)
{
    return PolyphaseIir2Designer::compute_nbr_coefs_from_proto(attenuation, transition);
}

double iir2designer_compute_atten_from_order_tbw(int nbr_coefs, double transition)
{
    return PolyphaseIir2Designer::compute_atten_from_order_tbw(nbr_coefs, transition);
} 

int iir2designer_compute_coefs(double* coef_arr, double attenuation, double transition)
{
    return PolyphaseIir2Designer::compute_coefs(coef_arr, attenuation, transition);
}

void iir2designer_compute_coefs_spec_order_tbw(double* coef_arr, int nbr_coefs, double transition)
{
    PolyphaseIir2Designer::compute_coefs_spec_order_tbw(coef_arr, nbr_coefs, transition);
}

	// inline double	PolyphaseIir2Designer::compute_phase_delay(double a, double f_fs)
	// inline double	PolyphaseIir2Designer::compute_group_delay(double a, double f_fs, bool ph_flag)
	// inline double	PolyphaseIir2Designer::compute_group_delay(const double coef_arr[], int nbr_coefs, double f_fs, bool ph_flag)
