import nlopt
import numpy as np
import json
import subprocess
import os
from distutils.dir_util import copy_tree
import shutil
import random
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# complementary logistic function
def compl_logistic(x, L=0.001, k=50.0, x0=0.0):

    fx = L - ( L / (1 + np.exp(-k* (x - x0))))
    return fx

# count number of carbon atoms in coord.xyz file
def count_carbon_atoms():
    n_line = 0
    nC = 0
    with open("coord.xyz") as f:
        for entire_line in f:
            n_line += 1
            if n_line >= 2:
                line = entire_line.strip()
                if "C " in line:
                    nC += 1
    return nC

def param_reader_LnEEQ(filepath):
    f = open("%s" % filepath, 'r')
    i=0
    param = np.zeros(fitpardim)
    for line in f:
        ls = line.strip()
        lss = np.asarray(ls.split())
        if len(lss) != fitpardim[1]:
            print("Reading parameter file went wrong! (dim1)")
            quit()
        param[i][:] = lss
        i += 1
    if i != fitpardim[0]:
        print("Reading parameter file went wrong! (dim2)")
    return param

def param_writer_LnEEQ(filepath, param):
    f = open("%s" % filepath, 'w')
    for i in range(np.shape(param)[0]):
        f.write("%10.6f    %10.6f    %10.6f    %10.6f\n" % 
                (param[i][0],param[i][1],param[i][2],param[i][3])) #@fixed1
    f.close()


# adjust EEQ parameters
def update_fitpar(x):
    param = param_reader_LnEEQ(param_filepath)
    for ix in param_fit_list:
        param[x_for_fit][ix] = x[ix]

    param_writer_LnEEQ(param_filepath, param)
    print("New params are:")
    print("%10.6f  %10.6f  %10.6f  %10.6f" % (x[0],x[1],x[2],x[3]))

# functions for running GFN-FF calculations
def run_gfnff(xtbver):
    # Check if the "coord" file exists in the current directory

    # Check if the "gfnff_topo" file exists and remove it
    if os.path.isfile("gfnff_topo"):
        os.remove("gfnff_topo")
    if os.path.isfile("gfnff_charges"):
        os.remove("gfnff_charges")
    try:
        # Run the binary and capture both stdout and stderr
        if not os.path.isfile("coord.xyz"):
            return None, "Error: 'coord.xyz' file not found in the current directory"
        result = subprocess.run(
            ["%s" % xtbver, "coord.xyz", "--gfnff"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Set to True for text output, False for binary
        )
        # Check if the process was successful (return code 0)
        if result.returncode == 0:
            return result.stdout, 0  # Successful, return stdout and no stderr
        else:
            return result.stdout, result.stderr  # Failed, return no stdout and stderr

    except Exception as e:
        print(e)
        return None, str(e)  # Return error message if an exception occurs


def get_eeq():
    if os.path.isfile("gfnff_charges"):
        with open("gfnff_charges") as f:
            nat=len(f.readlines())
        charge = np.zeros(nat)
        with open("gfnff_charges") as f:
            for i in range(nat):
                charge[i] = float(f.readline())
    else:
        print('No gfnff_charges file !!!')
        print(os.getcwd())
    return charge

def get_reference_charge():
    if os.path.isfile("hirshfeld.apc"):
        with open("hirshfeld.apc") as f:
            nat=len(f.readlines())
        charge = np.zeros(nat)
        with open("hirshfeld.apc") as f:
            for i in range(nat):
                charge[i] = float(f.readline())
    else:
        print('No reference charges file !!!')
        print(os.getcwd())
    return charge
         

def run_all_calcs():
    global fit_iter
    q_gfnff={}
    q_ref={}
    nsamp=-1

    # go through all the folders and subfolders
    x_folder = [d for d in os.listdir() if os.path.isdir(d)]
    for j, xdir in enumerate(x_folder):
        if xdir == ".git":
            continue
        os.chdir(xdir)
        subdirectories = [d for d in os.listdir() if os.path.isdir(d)]
        for i, d in enumerate(subdirectories):
            # check for sample folder and change directory
            if os.path.exists(d):
                os.chdir(d)
            elif os.path.exists(xdir + "/" + d):
                os.chdir(xdir)
                os.chdir(d)
            else:
                print("Error: No directory found for %s" % d)
                quit()
            # check for coord file and run gfnff
            if os.path.isfile("coord.xyz"):
                # run "xtb_acfit coord --gfnff" from ~/bin/
                # also deletes gfnff_topo beforehand
                output, error = run_gfnff("xtb_676a07_acfitEEQ")
                nsamp += 1
                if error != 0:
                    print("  Directory: %s" % d)
                    print("  Error: %s" % error)
                    # also print last 10 lines of stdout
                    print("  stdout:")
                    # split output by new lines and print the last 10 lines
                    print(output.splitlines()[-10:])
                    quit()
                # retrieve GFN-FF EEQ charge of Ln from gfnff_charges file
                q_gfnff["%s" % nsamp] = get_eeq()
                q_ref["%s" % nsamp] = get_reference_charge()
                os.chdir(geopath)
            else:
                print("Error: No coord file found in %s" % d)
                quit()
            # printout folders with deviation larger that 2.0
            if fit_iter == 1:
                q_dev = float(q_gfnff["%s" % nsamp][0]) - float(q_ref["%s" % nsamp][0])
                if abs(q_dev) >= 1.5:
                    print("  Sample %s has deviation of %s" % (d, q_dev))
    return q_gfnff, q_ref, nsamp


def plot_charges(q_gfnff, q_ref, fit_iter, nsamp, x, fx):
    # Get the number of samples
    nsamp = len(q_gfnff)
    global dir_str
    global x_for_fit
    # Initialize lists to store the charges
    gfnff_charges = []
    ref_charges = []
    plotpath = "/home/thor/Promotion/0_xtb_work_issues_here/xtb/_test/gfnff_high_cn/actinoid/2024_03_14_new_ac_fitset/fit_plots"
    
    # Iterate over the samples
    for i in range(1, nsamp):
        # Get the charges for the first atom in the sample
        gfnff_charge = q_gfnff[str(i)][0]
        ref_charge = q_ref[str(i)][0]
        
        # Append the charges to the lists
        gfnff_charges.append(gfnff_charge)
        ref_charges.append(ref_charge)
    
    # Plot the charges
    plt.plot([-4, 4], [-4, 4], color='grey', linestyle='-', linewidth=1)
    plt.scatter(ref_charges, gfnff_charges, color='blue', marker= '+')
    plt.grid()
    plt.text(-1.5, 3.5, "Error: %12.6f" % fx)
    plt.xlabel('Reference Charge')
    plt.ylabel('GFN-FF Charge')
    plt.xlim(-2, 4)
    plt.ylim(-2, 5)
    plt.title("Partial Charges for %s samples \n 1 param: %12.6f %12.6f %12.6f %12.6f\n 2 param: %12.6f %12.6f %12.6f %12.6f\n 3 param: %12.6f %12.6f %12.6f %12.6f" % (nsamp, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]), y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plotpath, 'gfnffEEQvsHirshfeld_%s.png' % (fit_iter)))
    plt.clf()

# Function takes current xparams and performs a quadratic interpolation using the parameters given by x_for_fit
# that is 4 parameters for each entry in x_for_fit
def quadratic_interpolation(xparams):
    # reshape xparams to a 4x3 matrix
    xparams_2D = np.reshape(xparams, (3, 4))
    # define zero np array with dimension 15x4 for all fit parameters
    fit_params = np.zeros((15, 4))
    # perform quadratic interpolation for each parameter
    for i in range(4):
        # Define x values
        x_values = np.array([i+1 for i in x_for_fit])
        # Define y values using xparams
        y_values = xparams_2D[:, i]
        # Define quadratic function
        def quadratic_func(x, a, b, c):
            return a * x**2 + b * x + c
        # Perform quadratic fitting
        coefficients = curve_fit(quadratic_func, x_values, y_values)
        # setup x values for interpolation
        x_interp = np.linspace(1, 15, 15)
        # calculate y values for interpolation
        y_interp = quadratic_func(x_interp, *coefficients[0])
        # save the interpolated values to fit_params
        for j in range(15):
            fit_params[j][i] = y_interp[j]
    print('Interpolated parameters:\n', fit_params)
    # update param_filepath with interpolated values
    param_writer_LnEEQ(param_filepath, fit_params)



# define the objective function
def f(x, grad):
    # start timing
    start_time = time.time()
    # increase fit_iter
    global fit_iter
    fit_iter += 1
    print("Iteration %s" % fit_iter)
    # update the fit parameters
    quadratic_interpolation(x)
    # define base path
    current_directory = os.getcwd()
    # define absolute path to calculations folder
    global geopath
    os.chdir(geopath)
    # run all calculations; also returns the GFN-FF Ln-charges for each sample
    q_gfnff, q_ref, nsamp = run_all_calcs()
    print("Number of samples: %s" % nsamp)
    os.chdir(geopath)
    # calculate the pearson correlation coefficient
    q_gfnff_actinide = [q[0] for q in q_gfnff.values()]
    q_ref_actinide = [q[0] for q in q_ref.values()]
    correlation = np.corrcoef(q_gfnff_actinide, q_ref_actinide)[0, 1]
    # assign the score
    q_SD_sum = 0.0 # sum of squared deviations of FF charges from reference
    q_AD_sum = 0.0 # sum of absolute deviations of FF charges from reference
    nsum = 0.0
    for i in range(nsamp):
        nat = len(q_gfnff["%s" % i])
        # iterate over all atoms
        for j in range(nat):
            qff  = float(q_gfnff["%s" % i][j]) # is a np array of size nat
            qref = float(q_ref["%s" % i][j])
            q_AD_sum += abs(qff - qref)
            q_SD_sum += (qff - qref)**2
            nsum += 1
    # calculate Mean Absolute Deviation
    q_MAD = q_AD_sum/nsum
    # calculate Mean Square Deviation from sum of SD
    q_MSD = q_SD_sum/nsum
    # take root for RMSD
    q_RMSD = np.sqrt(q_MSD)
    # choose error function
    i_err = 2
    if i_err == 0:
        corr_scale = 1.0
        if fit_iter == 1:
            print("Using MAD + %s - %s*abs(correlation) as error function." % (corr_scale, corr_scale) )    
        print("Correlation: %6.2f" % correlation, "  MAD: %12.6f" % q_MAD)
        fx = float(q_MAD) + corr_scale - corr_scale*abs(correlation) 
    elif i_err == 1:
        if fit_iter == 1:
            print("Using MAD as error function.")
        fx = float(q_MAD)
    elif i_err == 2:
        if fit_iter == 1:
            print("Using RMSD as error function.")
        fx = float(q_RMSD)
    elif i_err == 3:
        corr_scale = 2.0
        err_corr = corr_scale - corr_scale*abs(correlation)
        if fit_iter == 1:
            print("Using MSD + %s - %s*abs(correlation) as error function." % (corr_scale, corr_scale) )    
        print("1-Correlation: %7.3f" % err_corr, "  SD: %7.3f" % q_MSD)
        fx = float(q_MSD) + err_corr 
    else:
        print("Error: No error function chosen.")
        quit()
    # @thomas delete penalty for large cnf
    global init_cnf
    global fix_cnf
    if fix_cnf: # @thorm
        fx = fx + 0.05*abs(x[2] - init_cnf)
    # plot the charges
    if fit_iter % 10 == 0 or fit_iter == 1:
        plot_charges(q_gfnff, q_ref, fit_iter, nsamp, x, fx)
    print("Current dq(x)= %12.6f" % fx)
    print("Timing: %s" % (time.time() - start_time))
    print("")
    if fit_iter == 9999:
        print("Reached maximum number of iterations.")
        quit()
    return fx


#
def main():
    #############
    # Define global variables
    #############
    # absolute path to the parameter file list:
    #  dimension1=len(param_fit_list)
    #  dimension2=len(ln_fit_list)
    #quit() # delete line after xTB version is available
    global param_filepath
    param_filepath = "/home/thor/bin/gfnff_AcEEQ_param.txt" # @thomas change
    # define path to fit set folder
    global geopath
    geopath = "/home/thor/Promotion/0_xtb_work_issues_here/xtb/_test/gfnff_high_cn/actinoid/2024_03_14_new_ac_fitset/2024_04_11_fitset_reduced_noOutlier" # @thomas change
    # define parameter dimensions
    global fitpardim
    ndim1=15
    ndim2=4
    fitpardim=(ndim1, ndim2)
    # define which parameters should be fitted
    global param_fit_list
    param_fit_list = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11]) # is fixed to [0,1,2,3] !!!!@fixed1
    # define strings for Ln
    global dir_str
    dir_str = ["Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"] # @thomas change
    #dir_str = ["La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"] # @thomas change
    # define which Ln/Ac should be fitted ############################################
    # La=0, Ce=1, ..., Lu=15
    # Ac=0, Th=1, ..., Lr=15
    global x_for_fit
    x_for_fit = [0, 7, 14] # @thomas change @thorm
    global fix_cnf
    fix_cnf = False # @thomas change @thorm
    print("Fix cnf: %s" % fix_cnf)
    print("Fitting all EEQ parameters on Elements %s %s %s" % (dir_str[x_for_fit[0]],dir_str[x_for_fit[1]],dir_str[x_for_fit[2]]))
    # global fit iteration counter
    global fit_iter
    fit_iter = 0
    
    # check if dimensions fit the file
    fitparams = param_reader_LnEEQ(param_filepath)
    if np.shape(fitparams)[0] < len(param_fit_list):
        print("The read parameter file does not match the number of parameters you want to fit!")
        quit()
    
    print("Loaded initial parameters:")
    print("     chi         gam         cnf         alp")
    for i in range(np.shape(fitparams)[0]):
        print("%10.6f  %10.6f  %10.6f  %10.6f" % (fitparams[i][0],fitparams[i][1],
            fitparams[i][2],fitparams[i][3],))
    
    # user output
    print("")
    # define optimizer for fit
    nPar = len(param_fit_list)
    print("Number of parameters to fit: %s" % nPar)
    opt = nlopt.opt(nlopt.LN_BOBYQA, nPar)  # LN_BOBYQA is L="local" N="derivative-free"
    
    # load objective function to be minimized into opt
    opt.set_min_objective(f)
    
    # define parameters that are fitted ###########################################
    xparam = []
    for i in x_for_fit:
        xparam.extend(fitparams[i])
    print("xparam:", xparam)
    # update fit parameters after performing quadratic interpolation
    #quadratic_interpolation(xparam)
    # debugging: Fix x(2) by large penalty
    global init_cnf
    init_cnf = xparam[2]
    ### initial step size #########################################################
    ini_step = 0.05
    dx_ini = []
    for i in range(nPar):
        dx_ini.append(xparam[i]*ini_step)
    #dx_ini = xparam*ini_step
    opt.set_initial_step(dx_ini)
    ### error function convergence threshold ######################################
    fx_thr = 0.00001  #
    opt.set_stopval(fx_thr)
    ### change in parameter threshold #############################################
    dx_thr = 1.0e-7  #
    xtol_arr = dx_thr * np.ones(nPar)
    opt.set_xtol_abs(xtol_arr)
    
    # start optimization
    param_opt = opt.optimize(xparam)
    # print optimized parameters
    print(param_opt)
    quadratic_interpolation(param_opt)
    
    print('All done. Great success!')

if __name__ == "__main__":
    main()
