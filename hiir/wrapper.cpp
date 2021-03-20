#include "hiir/PolyphaseIir2Designer.h"

using namespace hiir;

extern "C"
{
    int iir2designer_compute_nbr_coefs_from_proto(double attenuation, double transition);
    double iir2designer_compute_atten_from_order_tbw(int nbr_coefs, double transition);
    int iir2designer_compute_coefs(double* coef_arr, double attenuation, double transition);
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
	return	PolyphaseIir2Designer::compute_coefs(coef_arr, attenuation, transition);
}

	// inline void	PolyphaseIir2Designer::compute_coefs_spec_order_tbw(double coef_arr[], int nbr_coefs, double transition)
	// inline double	PolyphaseIir2Designer::compute_phase_delay(double a, double f_fs)
	// inline double	PolyphaseIir2Designer::compute_group_delay(double a, double f_fs, bool ph_flag)
	// inline double	PolyphaseIir2Designer::compute_group_delay(const double coef_arr[], int nbr_coefs, double f_fs, bool ph_flag)
