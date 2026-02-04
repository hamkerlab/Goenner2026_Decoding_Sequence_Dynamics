# PROBLE WITH THE REGRESSION APPROACH:
# HETEROSCEDASTICITY!!!


# Alternative simplified approach:
# Even with heteroskedasticity, estimated coefficients are unbiased, while p-values are biased.
# Maybe compute a "modulation score" based on regression coefficients,
# e.g.:
# abs(mean sinusoidal coefficient) / abs(intercept)
#

# What about Welch's test (welch_anova_test):
# When binning phases, this could be used!

# How about another (complicated) yet different approach:
# In a linear mixed model, include spike count and phase as separate predictors for x-step?
# Problem: These predictors are highly correlated!

# Note:
# Normalizing xy-step estimates by a factor of 1/sqrt(n) or sqrt(n) doesn't work!
# Reason: We don't have a closed-form expression for the precise dependence of the estimation error on n.
# All we know is that for "small" numbers of spikes (n), the error behaves as 1/sqrt(n)

## Install package manager "pacman"
if(!require(pacman)) {install.packages("pacman")}

## List all necessary packages for your analysis here.
## pacman will install them if they aren't in your working environment.
pacman::p_load(pacman,
               Rcpp,
               Directional,
               car,
               RcppCNPy,
               Rfast,
               reticulate,
               zoo,
               stats,
               bayestestR,
               ggplot2,
               sandwich,
               lmtest,
               estimatr,
               gsignal,
               circular, 
               data.table,
               psych,
               patchwork,
               Hmisc)


if (!require(rstudioapi)) {
  install.packages("rstudioapi")
}

## get file path
current_filepath <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(current_filepath)

# Images directory for saving
imageDirectory <- '../latex/figs_r'

source_python("pytest.py")
#py_install("scipy")
py_require(c("scipy"))
source_python("py_hilbert.py")

# Data files (one level up, then into data/)
pkl <- read_pkl('../data/altspeed_1.0_sim_seq_paramsid_4_1000reps_1mincells_79steps_npformat.npy')
pkl_contin <- read_pkl('../data/sim_seq_paramsid_4_1000reps_1mincells_79steps_npformat.npy')
pkl_speedmod <- read_pkl('../data/speedmod_1.0_sim_seq_paramsid_4_1000reps_1mincells_79steps_npformat.npy')

# intervals for plotting
phase_breaks <- c(0, pi/4, pi/2, 3*pi/4, pi)
phase_labels <- c(
  expression(0), 
  expression(pi*phantom(0)/4), 
  expression(pi*phantom(0)/2), 
  expression(3*pi*phantom(0)/4), 
  expression(phantom("/")*pi) # force aligned x labels. friggin R
)

format_p_value <- function(p_value, min_val = 0.001, digits = 3) {
  if (p_value < min_val) {
    # format threshold in scientific notation if very small
    if (min_val >= 0.001) {
      threshold_str <- format(min_val, scientific = FALSE)
    } else {
      threshold_str <- format(min_val, scientific = TRUE)
    }
    return(paste("<", threshold_str))
  } else {
    return(paste("=", round(p_value, digits)))
  }
}

#source_python(paste(base_path, "Submissions/Step size decoding/Sequence_Decoding/R-scripts/py_hilbert.py", sep=''))
theme_set(theme_bw() + 
            theme(
              legend.position = "bottom",
              axis.text = element_text(size = 12),
              plot.title = element_text(face = "bold")
            ))

alt_x_true <- array( unlist(pkl[1]), dim=c(1000,79) ) # pkl[1] is of type list
alt_y_true <- array( unlist(pkl[2]), dim=c(1000,79) )
alt_x_mean <- array( unlist(pkl[3]), dim=c(1000,79) )
alt_y_mean <- array( unlist(pkl[4]), dim=c(1000,79) )
alt_totspikes <- array( unlist(pkl[5]), dim=c(1000,79) )
#alt_xdiff_true <- transpose( diff(transpose(alt_x_true), 1) ) # Ensure diff. is applied to correct dimension
alt_xdiff_true <- t( diff(t(alt_x_true), 1) ) # Ensure diff. is applied to correct dimension
alt_ydiff_true <- t( diff(t(alt_y_true), 1) )
alt_xydiff_true <- sqrt(alt_xdiff_true**2 + alt_ydiff_true**2)
alt_xdiff_mean <- t( diff(t(alt_x_mean), 1) ) 
alt_ydiff_mean <- t( diff(t(alt_y_mean), 1) ) 
alt_xydiff_mean <- sqrt(alt_xdiff_mean**2 + alt_ydiff_mean**2)
alt_totspikes_rollmean <- t(rollmean(t(alt_totspikes), 2)) # correct (no angles)

harmonic_mean_rolling <- function(x, k){
  rollapply(x, width=k, FUN=harmonic.mean, zero=TRUE)
}

alt_totspikes_harmonicmean <- t(harmonic_mean_rolling(t(alt_totspikes), 2)) # correct (no angles)

co_x_true <- array( unlist(pkl_contin[1]), dim=c(1000,79) ) # pkl[1] is of type list
co_y_true <- array( unlist(pkl_contin[2]), dim=c(1000,79) )
co_x_mean <- array( unlist(pkl_contin[3]), dim=c(1000,79) )
co_y_mean <- array( unlist(pkl_contin[4]), dim=c(1000,79) )
co_totspikes <- array( unlist(pkl_contin[5]), dim=c(1000,79) )
co_xdiff_true <- t( diff(t(co_x_true), 1) ) # Ensure diff. is applied to correct dimension
co_xdiff_true_rs <- array_reshape(co_xdiff_true, dim=c(78*1000, 1))
co_ydiff_true <- t( diff(t(co_y_true), 1) )
co_ydiff_true_rs <- array_reshape(co_ydiff_true, dim=c(78*1000, 1))
co_xydiff_true <- sqrt(co_xdiff_true**2 + co_ydiff_true**2)
co_xydiff_true_rs <- array_reshape(co_xydiff_true, dim=c(78*1000, 1))
co_xdiff_mean <- t( diff(t(co_x_mean), 1) ) 
co_xdiff_rs <- array_reshape(co_xdiff_mean, dim=c(78*1000, 1)) 
co_ydiff_mean <- t( diff(t(co_y_mean), 1) )
co_ydiff_rs <- array_reshape(co_ydiff_mean, dim=c(78*1000, 1)) 
co_xydiff_mean <- sqrt(co_xdiff_mean**2 + co_ydiff_mean**2)
co_xydiff_mean_rs <- array_reshape(co_xydiff_mean, dim=c(78*1000, 1))
co_totspikes_rollmean <- t(rollmean(t(co_totspikes), 2)) # correct (no angles)
co_totspikes_rollmean_rs <- array_reshape(co_totspikes_rollmean, dim=c(78*1000, 1)) 
co_totspikes_harmonicmean <- t(harmonic_mean_rolling(t(co_totspikes), 2)) # correct (no angles)
co_totspikes_harmonicmean_rs <- array_reshape(co_totspikes_harmonicmean, dim=c(78*1000, 1)) 

# co_theta_rs <- atan2(co_ydiff_rs, co_xdiff_rs)
# co_theta_true_rs <- atan2(co_ydiff_true_rs, co_xdiff_true_rs)
# co_Restim_x_rs <- co_xdiff_true_rs / cos(co_theta_rs)
# co_Restim_y_rs <- co_ydiff_true_rs / sin(co_theta_rs)
# co_Restim_xy_rs <- 0.5*(co_Restim_x_rs + co_Restim_y_rs)
# plot(co_totspikes_rollmean_rs[1:1000], abs(co_xydiff_true_rs[1:1000] - co_xydiff_mean_rs[1:1000]))
# plot(co_totspikes_rollmean_rs[1:1000], abs(co_xydiff_true_rs[1:1000] - co_Restim_x_rs[1:1000]))
# plot(co_totspikes_rollmean_rs[1:1000], abs(co_xydiff_true_rs[1:1000] - co_Restim_xy_rs[1:1000]))

sm_x_true <- array( unlist(pkl_speedmod[1]), dim=c(1000,79) ) # pkl[1] is of type list
sm_y_true <- array( unlist(pkl_speedmod[2]), dim=c(1000,79) )
sm_x_mean <- array( unlist(pkl_speedmod[3]), dim=c(1000,79) )
sm_y_mean <- array( unlist(pkl_speedmod[4]), dim=c(1000,79) )
sm_totspikes <- array( unlist(pkl_speedmod[5]), dim=c(1000,79) )
sm_xdiff_true <- t( diff(t(sm_x_true), 1) ) # Ensure diff. is applied to correct dimension
sm_xdiff_true_rs <- array_reshape(sm_xdiff_true, dim=c(78*1000, 1))
sm_ydiff_true <- t( diff(t(sm_y_true), 1) )
sm_xdiff_mean <- t( diff(t(sm_x_mean), 1) ) 
sm_xdiff_rs <- array_reshape(sm_xdiff_mean, dim=c(78*1000, 1)) 
sm_ydiff_mean <- t( diff(t(sm_y_mean), 1) ) 
sm_ydiff_rs <- array_reshape(sm_ydiff_mean, dim=c(78*1000, 1)) 
sm_xydiff_mean <- sqrt(sm_xdiff_mean**2 + sm_ydiff_mean**2)
sm_xydiff_mean_rs <- array_reshape(sm_xydiff_mean, dim=c(78*1000, 1)) 
sm_totspikes_rollmean <- t(rollmean(t(sm_totspikes), 2)) # Correct (no angles)
sm_totspikes_rollmean_rs <- array_reshape(sm_totspikes_rollmean, dim=c(78*1000, 1)) 
sm_ydiff_true_rs <- array_reshape(sm_ydiff_true, dim=c(78*1000, 1))
sm_xydiff_true <- sqrt(sm_xdiff_true**2 + sm_ydiff_true**2)
sm_xydiff_true_rs <- array_reshape(sm_xydiff_true, dim=c(78*1000, 1))

plot(alt_x_true[1,])
plot(alt_y_true[1,])
plot(alt_x_true[1,], alt_y_true[1,])
lines(alt_x_true[1,], alt_y_true[1,])

plot(alt_x_mean[1,], alt_y_mean[1,])
lines(alt_x_mean[1,], alt_y_mean[1,])

plot(co_x_mean[1,], co_y_mean[1,])
lines(co_x_mean[1,], co_y_mean[1,])

plot(sm_x_mean[1,], sm_y_mean[1,])
lines(sm_x_mean[1,], sm_y_mean[1,])

#plot(alt_totspikes[,], alt_x_mean[,]) # not very informative
#plot(alt_xdiff_mean, alt_xdiff_true) 
#plot(alt_xdiff_true, alt_xdiff_mean) # Variability of xdiff_mean increases for larger true step sizes
#plot(alt_xdiff_true[1,], alt_xdiff_mean[1,])


#plot(alt_totspikes_rollmean, alt_xdiff_true ) # Largest true x-step sizes for lowest spike counts
#plot(alt_totspikes_rollmean, alt_xdiff_mean ) # Largest x-step estimates for lowest spike counts
#plot(co_totspikes_rollmean, co_xdiff_mean ) # Flat shape with larger variability for low spike counts
#plot(sm_totspikes_rollmean, sm_xdiff_mean ) # Large variability for low spike counts, also higher mean x-step for low spike counts?

#plot(alt_totspikes_rollmean[1,], alt_xdiff_true[1,] ) # Largest true x-step sizes for lowest spike counts
#plot(alt_totspikes_rollmean[1,], alt_xdiff_mean[1,] ) # Largest x-step estimates for lowest spike counts
#plot(alt_totspikes_rollmean[1:100,], alt_xdiff_mean[1:100,] )
#plot(alt_totspikes_harmonicmean[1:100,], alt_xdiff_mean[1:100,] )

#plot(co_totspikes_rollmean[1:100,], co_xdiff_mean[1:100,] ) # Flat shape with larger variability for low spike counts
#plot(co_totspikes_harmonicmean[1:100,], co_xdiff_mean[1:100,] )

#plot(co_totspikes_rollmean[1:100,], co_xydiff_mean[1:100,] )
#plot(co_totspikes_harmonicmean[1:100,], co_xydiff_mean[1:100,] )

#plot(sm_totspikes_rollmean[1,], sm_xdiff_mean[1,] ) # Large variability for low spike counts, also higher mean x-step for low spike counts?



library(circular)
circmean_rolling <- function(x, k){
  atan2(rollmean(sin(x), k), rollmean(cos(x), k))
}




# Next step:
# Calculate phases based on the Hilbert transform! (Python, scipy.signal.hilbert )
source_python("py_hilbert.py")
#source_python("Z:/Submissions/Step size decoding/py_hilbert.py") # Works in BZW
htf <- hilb_transf(alt_totspikes)  # maps into [-pi/2,pi/2], artefacts at phase 0
htf <- hilb_transf(alt_totspikes - mean(alt_totspikes))  # maps into [-pi,pi]
#htf <- gsignal::hilbert(alt_totspikes - mean(alt_totspikes)) # Doesn't work as well?!
htf <- gsignal::hilbert(alt_totspikes) # Good!

indices <- 1:79
true_phase_vals <- indices / 79.0 * 5.2 * 2*pi

alt_hilb_phase <- Arg(htf)
#alt_hilb_phase <- t(matrix(rep(true_phase_vals, 1000), ncol = 1000))

#alt_phase_rollmean <- t(circmean_rolling(t(alt_hilb_phase), 2))  # THIS MUST BE A CIRCULAR MEAN!!! 
alt_phase_rollmean <- t(circmean_rolling(t(alt_hilb_phase), 2))


# TAG: Hilbert
cos_R_om_phi_offset <- function(params, t){
  params[1] * cos(params[2] * t + params[3]) + params[4]
}
squared_error <- function(params){
  sum( (cos_R_om_phi_offset(params, 1:79) - colmeans(co_totspikes))^2   )
}

res <- nlm(squared_error, c(30, 0.4, 0, 30))

#phi = atan2(lm_co_LFP_spikes$coefficients[3], lm_co_LFP_spikes$coefficients[2])
#plot(colmeans(co_totspikes))
#lines(colMins(co_totspikes, value=TRUE))
#lines(colMaxs(co_totspikes, value=TRUE))
#plot(colVars(co_totspikes))
#lines(cos_R_om_phi_offset(res$estimate, 1:79))

#co_htf <- hilb_transf(co_totspikes - mean(co_totspikes)) # maps into [-pi,pi] 
co_htf <- gsignal::hilbert(co_totspikes - mean(co_totspikes)) # maps into [-pi,pi] 
#co_htf <- hilb_transf(co_totspikes) # maps into [-pi/2, pi/2], artefacts
#co_hilb_phase <- Arg(co_htf)
co_hilb_phase <- t(matrix(rep(true_phase_vals, 1000), ncol = 1000)) # TEST
#co_phase_rollmean <- t(circmean_rolling(t(co_hilb_phase), 2))
co_phase_rollmean <- t(circmean_rolling(t(co_hilb_phase), 2))


#sm_htf <- hilb_transf(sm_totspikes - mean(sm_totspikes)) # maps into [-pi,pi]
sm_htf <- gsignal::hilbert(sm_totspikes - mean(sm_totspikes)) # maps into [-pi,pi]
#sm_htf <- hilb_transf(sm_totspikes) # maps into [-pi/2, pi/2], artefacts
#sm_hilb_phase <- Arg(sm_htf)
sm_hilb_phase <- t(matrix(rep(true_phase_vals, 1000), ncol = 1000))
#sm_phase_rollmean <- t(circmean_rolling(t(sm_hilb_phase), 2))
sm_phase_rollmean <- t(circmean_rolling(t(sm_hilb_phase), 2))


#hilb_ampli <- Mod(htf) # Not useful
plot(alt_totspikes[1,] - mean(alt_totspikes))
#plot(alt_totspikes[1,])
lines(20*cos(alt_hilb_phase[1,])) # Works!!! 


alt_xdiff_rs <- array_reshape(alt_xdiff_mean, dim=c(78*1000, 1)) # Is this correct?
alt_ydiff_rs <- array_reshape(alt_ydiff_mean, dim=c(78*1000, 1)) # Is this correct?
alt_xydiff_mean_rs <- array_reshape(alt_xydiff_mean, dim=c(78*1000, 1))
alt_xdiff_true_rs <- array_reshape(alt_xdiff_true, dim=c(78*1000, 1))
alt_xydiff_true_rs <- array_reshape(alt_xydiff_true, dim=c(78*1000, 1))
alt_totspikes_rollmean_rs <- array_reshape(alt_totspikes_rollmean, dim=c(78*1000, 1))
#alt_totspikes_rollmean_rs <- array_reshape(alt_totspikes_rollmean, dim=c(78*1000, 1))

alt_htf <- gsignal::hilbert(alt_totspikes - mean(alt_totspikes)) # maps into [-pi,pi]
alt_hilb_phase <- Arg(alt_htf)
alt_phase_rollmean <- t(circmean_rolling(t(alt_hilb_phase), 2))
alt_phase_rs <- array_reshape(alt_phase_rollmean, dim=c(78*1000, 1))

N <- 1000

df_alt <- as.data.frame(cbind(alt_xdiff_rs[1:N], alt_ydiff_rs[1:N], alt_phase_rs[1:N]))#, alt_totspikes_rollmean_rs)#, alt_xdiff_true_rs)#)


co_phase_rs <- array_reshape(co_phase_rollmean[1:1000,], dim=c(78*1000, 1))



#gp_xystep_vs_true_co <- ggplot(data = as.data.frame(co_xydiff_mean_rs, co_totspikes_rollmean_rs), aes(x=co_totspikes_rollmean_rs, y=co_xydiff_mean_rs)) +
N <- 5000
#gp_xystep_vs_true_co <- ggplot(data = as.data.frame(co_xydiff_mean_rs[1:N], co_xydiff_mean_rs[1:N]), aes(x=co_totspikes_rollmean_rs[1:N], y=co_xydiff_mean_rs[1:N])) +
#gp_xystep_vs_true_co <- ggplot(data = as.data.frame(co_xydiff_mean_rs, co_xydiff_mean_rs), aes(x=co_totspikes_rollmean_rs, y=co_xydiff_mean_rs)) +
df_plot <- as.data.frame(cbind(co_totspikes_rollmean_rs[1:N], co_xydiff_mean_rs[1:N]))
library(data.table)
setnames(df_plot, old=c('V1','V2'), new=c('co_totspikes_rollmean_rs', 'co_xydiff_mean_rs'))
gp_xystep_vs_true_co <- ggplot(data = df_plot, aes(x=co_totspikes_rollmean_rs, y=co_xydiff_mean_rs)) +    
  geom_point(alpha=0.4) +
  #geom_smooth() +
  geom_smooth(method = "loess", se=FALSE) + 
  xlab("Spike count") + ylab('Decoded xy-step' ) +
  ggtitle('Constant movement') +
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_xystep_vs_true_co)



N <- 5000 # 5000
p0 <- ggplot(data = as.data.frame(cbind(co_phase_rs[1:N], co_xdiff_rs[1:N])), aes(x=co_phase_rs[1:N], y=co_xdiff_rs[1:N])) +
  geom_point(color='purple',stat='identity', alpha=0.1)+ # alpha=0.01 for N > 5000  
  geom_smooth() +   
  coord_polar(start=pi/2, theta='x', clip='off', direction=-1) +
  #ylim(-4,20) + # (-4, NA  )
  ylim(-24,20) +  
  labs(title='Continuous movement')
p0

p0s <- ggplot(data = as.data.frame(cbind(co_phase_rs[1:N], co_totspikes_rollmean_rs[1:N])), aes(x=co_phase_rs[1:N], y=co_totspikes_rollmean_rs[1:N])) +
  geom_point(color='purple',stat='identity', alpha=0.1)+ # alpha=0.01 for N > 5000  
  geom_smooth() +   
  coord_polar(start=pi/2, theta='x', clip='off', direction=-1) +
  #ylim(-4,100) + # (-4, NA  )
  ylim(-44,100)   
  labs(title='Continuous movement')
p0s

#polar <- structure(list(degree = alt_phase_rs, 
#                        value = alt_xdiff_rs), 
#                   .Names = c("degree","value"), 
#                   class = "data.frame", 
#                   row.names = c(NA, -8L))

N <- 5000 # 5000
p1 <- ggplot(data = as.data.frame(cbind(alt_phase_rs[1:N], alt_xdiff_rs[1:N])), aes(x=alt_phase_rs[1:N], y=alt_xdiff_rs[1:N])) +
#p1 <- ggplot(polar, aes(x=degree, y=value)) +
  #geom_point(color='purple',stat='identity', alpha=0.01)+ # alpha=0.01 for N > 5000
  geom_point(color='purple',stat='identity', alpha=0.1)+ # alpha=0.01 for N > 5000  
  #geom_point(color='purple',stat='identity', aes(x=alt_phase_rs, y=alt_xdiff_rs), alpha=0.4)+
  geom_smooth() +   
  coord_polar(start=pi/2, theta='x', clip='off', direction=-1) +
  geom_segment(aes(x=0, xend=0, yend=0), arrow=arrow(type="closed")) +    # Vermutung: x und xend m?ssen gleich sein!
  #coord_polar(direction=-1) +  
  #geom_segment(aes(y=0, xend=3, yend=1))+
  #theme(axis.text.x =element_blank(),
  #      axis.text.y =element_blank())
  #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
  #ylim(-4,20) + # (-4, NA  )
  ylim(-24,20) +  
  labs(title='Plot title here')
  #scale_x_continuous(name="angle", limits=c(-3.14, 3.14)) # + 
#scale_y_continuous(name="Data", limits=c(-4,4))  
p1
imageFile <- file.path(imageDirectory,"x-step_vs_phase_alternating_polar.png")
ggsave(imageFile)

n_reps=1000 # 1000
co_xdiff_rs <- array_reshape(co_xdiff_mean[1:n_reps,], dim=c(78*n_reps, 1)) # Is this correct?
co_xdiff_true_rs <- array_reshape(co_xdiff_true[1:n_reps,], dim=c(78*n_reps, 1)) #
co_phase_rs <- array_reshape(co_phase_rollmean[1:n_reps,], dim=c(78*n_reps, 1))

df_co_xdiff_mean <- as.data.frame(co_xdiff_mean)
df_co_phase_rollmean <- as.data.frame(co_phase_rollmean)
# Required format:
# Column rep (=ID)
# Column xdiff
# Column phase

p2<- ggplot(data = as.data.frame(cbind(co_phase_rs[1:N], co_xdiff_rs[1:N])), aes(x=co_phase_rs[1:N], y=co_xdiff_rs[1:N])) +
  geom_point(color='purple',stat='identity', alpha=0.1)+ # alpha=0.4
  #geom_point(color='purple',stat='identity', aes(x=alt_phase_rs, y=alt_xdiff_rs), alpha=0.4)+
  geom_smooth() +   
  #coord_polar(start =-pi/2, theta='x', clip='off') +
  geom_segment(aes(y=0, xend=180, yend=4))+
  #theme(axis.text.x =element_blank(),
  #      axis.text.y =element_blank())
  #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
  ylim(-1,NA) + # -4, NA  
  scale_x_continuous(name="angle", limits=c(-3.14, 3.14)) # + 
#scale_y_continuous(name="Data", limits=c(-4,4))  
p2
imageFile <- file.path(imageDirectory,"x-step_vs_phase_alternating_constant.png")
ggsave(imageFile)

plot(co_phase_rs[1:N],co_totspikes_rollmean_rs[1:N])




#lm_co_spikes_sin_cos <- lm(co_totspikes_rollmean[1,] ~ cos(co_phase_rollmean)[1,] + sin(co_phase_rollmean)[1,])
lm_co_spikes_sin_cos <- lm(co_totspikes_rollmean_rs[1:N] ~ cos(co_phase_rs[1:N]) + sin(co_phase_rs[1:N]))
summary(lm_co_spikes_sin_cos)

p2s<- ggplot(data = as.data.frame(cbind(co_phase_rs[1:N], co_totspikes_rollmean_rs[1:N])), aes(x=co_phase_rs[1:N], y=co_totspikes_rollmean_rs[1:N])) +
  geom_point(color='purple',stat='identity', alpha=0.1)+ # alpha=0.4
  geom_line(aes(y=predict(lm_co_spikes_sin_cos))) + 
  geom_line(aes(y=lm_co_spikes_sin_cos$coefficients[1]/sqrt(predict(lm_co_spikes_sin_cos))), color='blue') +   
  #geom_smooth() +   
  #coord_polar(start =-pi/2, theta='x', clip='off') +
  geom_segment(aes(y=0, xend=180, yend=4))+
  #ylim(-1,NA) + # -4, NA  
  scale_x_continuous(name="angle", limits=c(-3.14, 3.14)) # + 
p2s
# TAG: Spikes vs. Phase

p3<- ggplot(data = as.data.frame(cbind(co_phase_rs[1:N], co_xydiff_mean_rs[1:N])), aes(x=co_phase_rs[1:N], y=co_xydiff_mean_rs[1:N])) +
  geom_point(color='purple',stat='identity', alpha=0.1)+ # alpha=0.4
  #geom_point(color='purple',stat='identity', aes(x=alt_phase_rs, y=alt_xdiff_rs), alpha=0.4)+
  geom_smooth() +   
  coord_polar(start =-pi/2, theta='x', clip='off') +
  geom_segment(aes(y=0, xend=180, yend=4))+
  #theme(axis.text.x =element_blank(),
  #      axis.text.y =element_blank())
  #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
  #ylim(-1,NA) + # -4, NA  
  ylim(-20,NA) +  
  scale_x_continuous(name="angle", limits=c(-3.14, 3.14)) # + 
#scale_y_continuous(name="Data", limits=c(-4,4))  
p3
imageFile <- file.path(imageDirectory,"xy-step_decoded_vs_phase_alternating_constant.png")
ggsave(imageFile)

co_xydiff_mean_rs_normalized <- 1.0 / sqrt(co_totspikes_rollmean_rs) * co_xydiff_mean_rs

#p3_norm<- ggplot(data = as.data.frame(co_phase_rs, co_xydiff_mean_rs_normalized), aes(x=co_phase_rs, y= co_xydiff_mean_rs_normalized)) +
#  geom_point(color='purple',stat='identity', alpha=0.1)+ # alpha=0.4
#  #geom_point(color='purple',stat='identity', aes(x=alt_phase_rs, y=alt_xdiff_rs), alpha=0.4)+
#  geom_smooth() +   
#  coord_polar(start =-pi/2, theta='x', clip='off') +
#  geom_segment(aes(y=0, xend=180, yend=4))+
#  #theme(axis.text.x =element_blank(),
#  #      axis.text.y =element_blank())
#  #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
#  ylim(-1,NA) + # -4, NA  
#  scale_x_continuous(name="angle", limits=c(-3.14, 3.14)) # + 
##scale_y_continuous(name="Data", limits=c(-4,4))  
#p3_norm




lm_xco_sin_cos1 <- lm(co_xdiff_mean[1,] ~ cos(co_phase_rollmean)[1,] + sin(co_phase_rollmean)[1,])
#lm_xco_linear <- lm(co_xdiff_mean[1,] ~ co_phase_rollmean[1,])
lm_xco_linear <- lm(co_xdiff_mean[1,] ~ 1)
waldtest(lm_xco_sin_cos1, vcov = vcovHC(lm_xco_sin_cos1, type = "HC0"))
#waldtest(lm_xco_linear, vcov = vcovHC(lm_xco_linear, type = "HC0"))

#df_co_xdiff_mean

#plot(abs(co_phase_rs), co_xdiff_rs)
plot(abs(co_phase_rs[1:77,]), co_xdiff_rs[1:77,])
#plot(co_phase_rs, co_xdiff_rs)
plot(co_phase_rs[1:77,], co_xdiff_rs[1:77,])

lm_xcors_sin_cos <- lm(co_xdiff_rs ~ cos(co_phase_rs) + sin(co_phase_rs))
lm_xcors_linear <- lm(co_xdiff_rs ~ co_phase_rs)
#lm_xcors_const <- lm(co_xdiff_rs ~ 1) # not very meaningful

coeftest(lm_xcors_linear, vcov = vcovHC(lm_xcors_linear, type = "HC0"))
coeftest(lm_xcors_sin_cos, vcov = vcovHC(lm_xcors_sin_cos, type = "HC0")) # ???
#waldtest(lm_xcors_linear, lm_xcors_sin_cos, vcov = vcovHC(lm_xcors_sin_cos, type = "HC0"))
waldtest(lm_xcors_sin_cos, vcov = vcovHC(lm_xcors_sin_cos, type = "HC0"))
waldtest(lm_xcors_linear, vcov = vcovHC(lm_xcors_linear, type = "HC0"))


library(estimatr)
lmrob_xcors_sin_cos <- lm_robust(co_xdiff_rs ~ cos(co_phase_rs) + sin(co_phase_rs))
lmrob_xcors_linear <- lm_robust(co_xdiff_rs ~ co_phase_rs)
summary(lmrob_xcors_sin_cos) # ???
summary(lmrob_xcors_linear)
waldtest(lmrob_xcors_sin_cos)
waldtest(lmrob_xcors_linear)


gls_xcors_sin_cos <- nlme::gls(co_xdiff_rs ~ cos(co_phase_rs) + sin(co_phase_rs))
gls_xcors_cos <- nlme::gls(co_xdiff_rs ~ cos(co_phase_rs))
gls_xcors_const <- nlme::gls(co_xdiff_rs ~ 1)
summary(gls_xcors_sin_cos)
summary(gls_xcors_cos)
summary(gls_xcors_const)
anova(gls_xcors_sin_cos, gls_xcors_const)
anova(gls_xcors_cos, gls_xcors_const)

#logLik(lm_xcors_sin_cos)
circlin.cor(co_phase_rs, co_xdiff_rs) # R^2 = 0.015, p=0 (?)
BIC(lm_xcors_sin_cos)
BIC(lm_xcors_linear)
#BIC(lm_xcors_const)
#bic_to_bf(BIC(lm_xcors_sin_cos), BIC(lm_xcors_const), log = TRUE) # ???

summary(lm_xcors_sin_cos) # R-squared=0.013, p < 2.2e-16 ?!
summary(lm_xcors_linear) # R-squared = 0.008, p < 2.2e-16
#summary(lm_xcors_const)
anova(lm_xcors_sin_cos, lm_xcors_linear) # Sig. difference?!
#plot(co_phase_rs, co_xdiff_rs)
plot(co_phase_rs[1,], co_xdiff_rs[1,])
circlin.cor(co_phase_rs, co_xdiff_rs) # Very small R-squared = 0.015

N <- 1000
df_co <- as.data.frame(cbind(co_xdiff_rs[1:N], co_phase_rs[1:N]))#, co_xdiff_true_rs, co_totspikes_rollmean_rs)
df_co2 <- as.data.frame(cbind(co_xdiff_rs[1:N], co_totspikes_rollmean_rs[1:N]))


pval_from_corrcoef <- function(rho, n){
  t <- rho * sqrt(n - 2) / sqrt(1 - rho**2)  
  p <- 2*pt(t, n-2, lower.tail = TRUE) # two-sided
  return(c(t, p))
}

circlin_cor_spearman <- function(x_angle, y, radians=TRUE){
  rho <- circlin.cor(rank(x_angle), rank(y))
}


cor_x_co <- cor.test(abs(co_phase_rs[1:N]), co_xdiff_rs[1:N], method='spearman') #
#cor_x_co <- circlin_cor_spearman(abs(co_phase_rs[1:N]), co_xdiff_rs[1:N])
gp_dots_co <- ggplot(data = df_co, aes(x=abs(co_phase_rs[1:N]), y=co_xdiff_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth() + #method = "loess", se=FALSE  #'lm'
  #stat_summary(fun = mean, geom = "point", shape=23, size=2, color='white') + 
  #stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +   
  geom_smooth(aes(y = co_xdiff_true_rs[1:N]), method = "loess", 
              linetype = "dashed", color = "red", linewidth = 2, se = FALSE) +
  #geom_hline(yintercept=mean(co_xdiff_true_rs), linetype="dashed", color = "red", linewidth=2) +    
  ylab("Decoded x-step") + xlab('Absolute Phase [rad]' ) +
  ggtitle('Smooth movement') +
  scale_x_continuous(
    breaks = phase_breaks,
    labels = phase_labels,
    limits = c(0, pi)
  ) +
  ylim(NA, 30) +  
  annotate("text", 
           label = paste("ρ =", round(cor_x_co$estimate, 2), 
                         ', p', format_p_value(cor_x_co$p.value, min_val = 1e-10)), 
           x = 0.8, y = 25, size = 7, colour = "black") +
  #annotate("text", label = paste('rho_cl =', round(cor_x_co[1,1], 3), ', p =', round(cor_x_co[1,2],3)), x = 1, y = 10, size = 8, colour = "black" ) +         
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=16),
        axis.text.y = element_text(size=16),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_co)
imageFile <- file.path(imageDirectory,"x-step_vs_phase_constant.png")
ggsave(imageFile)

# gp_dots_co_polar <- ggplot(data = df_co, aes(x=abs(co_phase_rs)[1:N], y=co_xdiff_rs[1:N])) +
#   geom_point(alpha=0.4) +
#   #coord_polar(start =-pi/2, theta='x', clip='off') +  
#   geom_smooth() + #method = "loess", se=FALSE  #'lm'
#   #stat_summary(fun = mean, geom = "point", shape=23, size=2, color='white') + 
#   #stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +   
#   ylab("x-step") + xlab('Phase' ) +
#   ggtitle('Constant movement') +
#   theme(plot.title=element_text(hjust=0.5) ,
#         axis.text.x = element_text(face="bold", size=14),
#         axis.text.y = element_text(face="bold", size=14),
#         title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
#   theme(legend.text = element_text(size = 20)) 
# plot(gp_dots_co_polar)

cor_xy_co <- cor.test(co_totspikes_rollmean_rs[1:N], co_xydiff_mean_rs[1:N], method='spearman') #
gp_dots_xy_co <- ggplot(data = df_co2, aes(x=co_totspikes_rollmean_rs[1:N], y=co_xydiff_mean_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  geom_hline(yintercept=mean(co_xydiff_true_rs), linetype="dashed", color = "red", linewidth=2) +    
  xlab("Spike count") + ylab('Decoded xy-step' ) +
  ggtitle('Smooth movement') +
  annotate("text", label = paste('rho =', round(cor_xy_co$estimate, 2), ', p =', round(cor_xy_co$p.value,3)), x = 50, y = 10, size = 8, colour = "black" ) +     
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_xy_co)
imageFile <- file.path(imageDirectory,"Decoded_xy-step_vs_spikes_constant.png")
ggsave(imageFile)


cor_xy_co_phase <- cor.test(abs(co_phase_rs)[1:N], co_xydiff_mean_rs[1:N], method='spearman')
#cor_xy_co_phase <- circlin_cor_spearman(abs(co_phase_rs[1:N]), co_xydiff_mean_rs[1:N]) 
gp_dots_xy_co_phase <- ggplot(data = df_co2, aes(x=abs(co_phase_rs)[1:N], y=co_xydiff_mean_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  ylab("Decoded xy-step") + xlab('Absolute Phase [rad]' ) +
  geom_smooth(aes(y = co_xydiff_true_rs[1:N]), method = "loess", 
              linetype = "dashed", color = "red", linewidth = 2, se = FALSE) +
  #geom_hline(yintercept=mean(co_xydiff_true_rs), linetype="dashed", color = "red", linewidth=2) +  
  ggtitle('Smooth movement') +
  scale_x_continuous(
    breaks = phase_breaks,
    labels = phase_labels,
    limits = c(0, pi)
  ) +
  ylim(NA, 30) +
  annotate("text", 
           label = paste("ρ =", round(cor_xy_co_phase$estimate, 2), 
                         ', p', format_p_value(cor_xy_co_phase$p.value, min_val = 1e-10)), 
           x = 0.8, y = 27, size = 7, colour = "black") +
  #annotate("text", label = paste('rho_cl =', round(cor_xy_co_phase[1,1], 3), ', p =', round(cor_xy_co_phase[1,2],3)), x = 1, y = 10, size = 8, colour = "black" ) +             
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=16),
        axis.text.y = element_text(size=16),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_xy_co_phase)

imageFile <- file.path(imageDirectory,"Decoded_xy-step_vs_phase_constant.png")
ggsave(imageFile)


cor_xy_alt_phase <- cor.test(abs(alt_phase_rs)[1:N], alt_xydiff_mean_rs[1:N], method='spearman')
#cor_xy_alt_phase <- circlin_cor_spearman(abs(alt_phase_rs[1:N]), alt_xydiff_mean_rs[1:N]) 
gp_dots_xy_alt_phase <- ggplot(data = df_alt, aes(x=abs(alt_phase_rs)[1:N], y=alt_xydiff_mean_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  ylab("Decoded xy-step") + xlab('Absolute Phase [rad]' ) +
  ggtitle('Alternating movement') +
  annotate("text", label = paste('rho =', round(cor_xy_alt_phase$estimate, 2), ', p =', round(cor_xy_alt_phase$p.value,3)), x = 1, y = 10, size = 8, colour = "black" ) +     
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_xy_alt_phase)
imageFile <- file.path(imageDirectory,"Decoded_xy-step_vs_phase_alternating.png")
ggsave(imageFile)


cor_co_phase_spikes <- cor.test(abs(co_phase_rs)[1:N], co_totspikes_rollmean_rs[1:N], method='spearman')
gp_dots_co_phase_spikes <- ggplot(data = as.data.frame(cbind(co_phase_rs[1:N],co_totspikes_rollmean_rs[1:N])), aes(x=abs(co_phase_rs)[1:N], y=co_totspikes_rollmean_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  xlab("abs(Phase)") + ylab('Spikes' ) +
  ggtitle('Smooth movement') +
  annotate("text", label = paste('rho =', round(cor_co_phase_spikes$estimate, 2), ', p =', round(cor_co_phase_spikes$p.value,3)), x = 1, y = 10, size = 8, colour = "black" ) +     
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_co_phase_spikes)
imageFile <- file.path(imageDirectory,"Spikes_vs_phase_constant.png")
ggsave(imageFile)

# CAUTION:
# It's POINTLESS to use partial correlation in our example!
# Reason: Phase and spikes are perfectly correlated!
# Nothing will remain after partialing out the spike.-to-phase corrrelation!
# co_cor_xdiff_absphase <- cor.test(abs(co_phase_rs)[1:N], co_xdiff_rs[1:N], method='spearman')
# co_cor_spikes_absphase <- cor.test(co_totspikes_rollmean_rs[1:N], abs(co_phase_rs[1:N]), method='spearman')
# co_cor_xdiff_spikes <- cor.test(co_totspikes_rollmean_rs[1:N], co_xdiff_rs[1:N], method='spearman')
# co_pcor_xdiff_absphase_partialout_spikes <- {(co_cor_xdiff_absphase$estimate - co_cor_spikes_absphase$estimate * co_cor_xdiff_spikes$estimate) / (sqrt(1 - co_cor_spikes_absphase$estimate^2) * sqrt(1 - co_cor_xdiff_spikes$estimate^2))
#                                              }
# print(co_pcor_xdiff_absphase_partialout_spikes)
# co_pcor_pval <- pval_from_corrcoef(as.numeric(co_pcor_xdiff_absphase_partialout_spikes), length(co_phase_rollmean[1,]))
# print(co_pcor_pval)
# 
# co_cor_xydiff_absphase <- cor.test(abs(co_phase_rs)[1:N], co_xydiff_mean_rs[1:N], method='spearman')
# co_cor_spikes_absphase <- cor.test(co_totspikes_rollmean_rs[1:N], abs(co_phase_rs[1:N]), method='spearman')
# co_cor_xydiff_spikes <- cor.test(co_totspikes_rollmean_rs[1:N], co_xydiff_mean_rs[1:N], method='spearman')
# co_pcor_xydiff_absphase_partialout_spikes <- {(co_cor_xydiff_absphase$estimate - co_cor_spikes_absphase$estimate * co_cor_xydiff_spikes$estimate) / (sqrt(1 - co_cor_spikes_absphase$estimate^2) * sqrt(1 - co_cor_xydiff_spikes$estimate^2))
# }
# print(co_pcor_xydiff_absphase_partialout_spikes)
# co_pcor_xy_pval <- pval_from_corrcoef(co_pcor_xydiff_absphase_partialout_spikes, length(co_phase_rollmean[1,]))
# print(co_pcor_xy_pval)



cor_co <- cor.test(co_totspikes_rollmean_rs[1:N], co_xdiff_rs[1:N], method='spearman') # rho=-0.03, p=0.33 - GOOD!
cor.test(co_totspikes_rollmean_rs, co_xdiff_rs, method='spearman') # rho=-0.06, p=2.2e-16 - GOOD ENOUGH!
gp_dots_vs_true_co <- ggplot(data = df_co2, aes(x=co_totspikes_rollmean_rs[1:N], y=co_xdiff_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  geom_hline(yintercept=mean(co_xdiff_true_rs), linetype="dashed", color = "red", linewidth=2) +    
  #stat_summary(fun = mean, geom = "bar", position="dodge") +  
  #stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +       xlab("Spike count") + ylab('Decoded x-step' ) +
  ggtitle('Smooth movement') +
  annotate("text", label = paste('rho =', round(cor_co$estimate, 2), ', p =', round(cor_co$p.value,3)), x = 50, y = 10, size = 8, colour = "black" ) +     
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_vs_true_co)
imageFile <- file.path(imageDirectory,"Decoded_x-step_vs_spikes_constant.png")
ggsave(imageFile)



sm_phase_rs <- array_reshape(sm_phase_rollmean[1:1000,], dim=c(78*1000, 1))
df_sm <- as.data.frame(cbind(sm_xdiff_rs[1:N], sm_ydiff_rs[1:N], sm_phase_rs[1:N]))

cor_sm_phase_spikes <- cor.test(abs(sm_phase_rs)[1:N], sm_totspikes_rollmean_rs[1:N], method='spearman')
gp_dots_sm_phase_spikes <- ggplot(data = as.data.frame(cbind(sm_phase_rs[1:N],sm_totspikes_rollmean_rs[1:N])), aes(x=abs(sm_phase_rs)[1:N], y=sm_totspikes_rollmean_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  xlab("abs(Phase)") + ylab('Spikes' ) +
  ggtitle('Phase-locked movement') +
  annotate("text", label = paste('rho =', round(cor_sm_phase_spikes$estimate, 2), ', p =', round(cor_sm_phase_spikes$p.value,3)), x = 1, y = 10, size = 8, colour = "black" ) +     
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_sm_phase_spikes)
imageFile <- file.path(imageDirectory,"Spikes_vs_phase_speedmod.png")
ggsave(imageFile)

cor_xy_sm_phase <- cor.test(abs(sm_phase_rs)[1:N], sm_xydiff_mean_rs[1:N], method='spearman')
gp_dots_xystep_sm <- ggplot(data = df_sm, aes(x=abs(sm_phase_rs[1:N]), y=sm_xydiff_mean_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(aes(y = sm_xydiff_true_rs[1:N]), method = "loess", 
              linetype = "dashed", color = "red", linewidth = 2, se = FALSE) +
  geom_smooth(method='lm') + #method = "loess", se=FALSE  #'lm'
  #stat_summary(fun = mean, geom = "point", shape=23, size=2, color='white') + 
  #stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +   
  ylab("Decoded xy-step") + xlab('Absolute Phase [rad]' ) +
  ggtitle('Phase-locked movement') +
  scale_x_continuous(
    breaks = phase_breaks,
    labels = phase_labels,
    limits = c(0, pi)
  ) +
  ylim(NA, 30) +
  annotate("text", 
           label = paste("ρ =", round(cor_xy_sm_phase$estimate, 2), 
                         ', p', format_p_value(cor_xy_sm_phase$p.value, min_val = 1e-10)), 
           x = 0.8, y = 27, size = 7, colour = "black") +
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=16),
        axis.text.y = element_text(size=16),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_xystep_sm)
imageFile <- file.path(imageDirectory,"Decoded_xy-step_vs_phase_speed-modulated.png")
ggsave(imageFile)


cor_sm_xdiff <- cor.test(abs(sm_phase_rs[1:N]), sm_xdiff_rs[1:N], method='spearman') # rho=-0.37, p=2.2e-16 - GOOD!
gp_dots_sm <- ggplot(data = df_sm, aes(x=abs(sm_phase_rs[1:N]), y=sm_xdiff_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(aes(y = sm_xdiff_true_rs[1:N]), method = "loess", 
              linetype = "dashed", color = "red", linewidth = 2, se = FALSE) +
  geom_smooth(method='lm') + #method = "loess", se=FALSE  #'lm'
  #stat_summary(fun = mean, geom = "point", shape=23, size=2, color='white') + 
  #stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +   
  ylab("Decoded x-step") + xlab('Absolute Phase [rad]' ) +
  ggtitle('Phase-locked movement') +
  scale_x_continuous(
    breaks = phase_breaks,
    labels = phase_labels,
    limits = c(0, pi)
  ) +
  ylim(NA, 30) +
  annotate("text", 
           label = paste("ρ =", round(cor_sm_xdiff$estimate, 2), 
                         ', p', format_p_value(cor_sm_xdiff$p.value, min_val = 1e-10)), 
           x = 0.8, y = 25, size = 7, colour = "black") +
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=16),
        axis.text.y = element_text(size=16),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_sm)
imageFile <- file.path(imageDirectory,"Decoded_x-step_vs_phase_speed-modulated.png")
ggsave(imageFile)


cor_alt_xdiff <- cor.test(abs(alt_phase_rs[1:N]), alt_xdiff_rs[1:N], method='spearman') # rho=-0.37, p=2.2e-16 - GOOD!
gp_dots_alt <- ggplot(data = df_alt, aes(x=abs(alt_phase_rs[1:N]), y=alt_xdiff_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') + #method = "loess", se=FALSE  #'lm'
  #stat_summary(fun = mean, geom = "point", shape=23, size=2, color='white') + 
  #stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +   
  ylab("x-step") + xlab('abs(Phase)' ) +
  ggtitle('Alternating movement') +
  annotate("text", label = paste('rho =', round(cor_alt_xdiff$estimate, 2), ', p =', round(cor_alt_xdiff$p.value,82)), x = 1, y = 10, size = 8, colour = "black" ) +         
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=16),
        axis.text.y = element_text(face="bold", size=16),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_alt)
imageFile <- file.path(imageDirectory,"Decoded_x-step_vs_phase_alternating.png")
ggsave(imageFile)




cor_sm <- cor.test(sm_totspikes_rollmean_rs[1:N], sm_xdiff_rs[1:N], method='spearman') # rho=-0.37, p=2.2e-16 - GOOD!
cor.test(sm_totspikes_rollmean_rs, sm_xdiff_rs, method='spearman') # rho=-0.35, p=2.2e-16 - GOOD!
gp_dots_vs_true_sm <- ggplot(data = df_sm, aes(x=sm_totspikes_rollmean_rs[1:N], y=sm_xdiff_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  xlab("Spike count") + ylab('Decoded x-step' ) +
  ggtitle('Speed-modulated movement') +
  annotate("text", label = paste('rho =', round(cor_sm$estimate, 2), ', p =', round(cor_sm$p.value,33)), x = 60, y = 10, size = 8, colour = "black" ) +       
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_vs_true_sm)
imageFile <- file.path(imageDirectory,"Decoded_x-step_vs_spikes_speed-modulated.png")
ggsave(imageFile)

cor_alt <- cor.test(alt_totspikes_rollmean_rs[1:N], alt_xdiff_rs[1:N], method='spearman') # rho=-0.37, p=2.2e-16 - GOOD!
cor.test(alt_totspikes_rollmean_rs, alt_xdiff_rs, method='spearman') # rho=-0.35, p=2.2e-16 - GOOD!
gp_dots_vs_true_alt <- ggplot(data = df_alt, aes(x=alt_totspikes_rollmean_rs[1:N], y=alt_xdiff_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  #geom_smooth() +  
  xlab("Spike count") + ylab('Decoded x-step' ) +
  ggtitle('Alternating movement') +
  annotate("text", label = paste('rho =', round(cor_alt$estimate, 2), ', p =', round(cor_alt$p.value,33)), x = 35, y = 10, size = 8, colour = "black" ) +       
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_vs_true_alt)
imageFile <- file.path(imageDirectory,"Decoded_x-step_vs_spikes_alternating.png")
ggsave(imageFile)


cor_sm_xydiff <- cor.test(sm_totspikes_rollmean_rs[1:N], sm_xydiff_mean_rs[1:N], method='spearman') # rho=-0.37, p=2.2e-16 - GOOD!
gp_dots_vs_true_sm_xydiff <- ggplot(data = df_sm, aes(x=sm_totspikes_rollmean_rs[1:N], y=sm_xydiff_mean_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  xlab("Spike count") + ylab('Decoded xy-step' ) +
  ggtitle('Speed-modulated movement') +
  annotate("text", label = paste('rho =', round(cor_sm_xydiff$estimate, 2), ', p =', round(cor_sm_xydiff$p.value,33)), x = 60, y = 10, size = 8, colour = "black" ) +       
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_vs_true_sm_xydiff)
imageFile <- file.path(imageDirectory,"Decoded_xy-step_vs_spikes_speed-modulated.png")
ggsave(imageFile)

cor_alt_xydiff <- cor.test(alt_totspikes_rollmean_rs[1:N], alt_xydiff_mean_rs[1:N], method='spearman') # rho=-0.37, p=2.2e-16 - GOOD!
gp_dots_vs_true_alt_xydiff <- ggplot(data = df_alt, aes(x=alt_totspikes_rollmean_rs[1:N], y=alt_xydiff_mean_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  xlab("Spike count") + ylab('Decoded xy-step' ) +
  ggtitle('Alternating movement') +
  annotate("text", label = paste('rho =', round(cor_alt_xydiff$estimate, 2), ', p =', round(cor_alt_xydiff$p.value,33)), x = 35, y = 10, size = 8, colour = "black" ) +       
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_vs_true_alt_xydiff)
imageFile <- file.path(imageDirectory,"Decoded_xy-step_vs_spikes_alternating.png")
ggsave(imageFile)


cor_alt <- cor.test(alt_totspikes_rollmean_rs[1:N], sm_xdiff_rs[1:N], method='spearman') # rho=-0.38, p=2.2e-16 - GOOD!
cor.test(alt_totspikes_rollmean_rs, sm_xdiff_rs, method='spearman') # rho=-0.38, p=2.2e-16 - GOOD!
gp_dots_vs_true_alt <- ggplot(data = df_alt, aes(x=alt_totspikes_rollmean_rs[1:N], y=alt_xdiff_rs[1:N])) +
  geom_point(alpha=0.4) +
  geom_smooth(method='lm') +
  xlab("Spike count") + ylab('Decoded x-step' ) +
  ggtitle('Alternating movement') +
  annotate("text", label = paste('rho =', round(cor_alt$estimate, 2), ', p =', round(cor_alt$p.value,36)), x = 30, y = 10, size = 6, colour = "black" ) +         
  theme(plot.title=element_text(hjust=0.5) ,
        axis.text.x = element_text(face="bold", size=14),
        axis.text.y = element_text(face="bold", size=14),
        title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
  theme(legend.text = element_text(size = 20)) 
plot(gp_dots_vs_true_alt)

library(patchwork)
#wrap_plots(p1, p2, p3, p4, ncol = 2, nrow = 2, widths = c(1, 0.5), heights = c(0.5, 1))

#gp_dots_vs_true_co + gp_dots_sm + gp_dots_xy_co + gp_dots_xystep_sm
gp_dots_vs_true_sm_xydiff + gp_dots_xy_co + gp_dots_vs_true_alt_xydiff + gp_dots_vs_true_sm  +  gp_dots_vs_true_co + gp_dots_vs_true_alt
imageFile <- file.path(imageDirectory,"Overview_xy-step_and_x-step_vs_spikes.png")
ggsave(imageFile)

gp_dots_xystep_sm + gp_dots_xy_co_phase + gp_dots_xy_alt_phase + gp_dots_sm + gp_dots_co + gp_dots_alt
imageFile <- file.path(imageDirectory,"Overview_xy-step_and_x-step_vs_phase.png")
ggsave(imageFile)



p1_co <- ggplot(data = df_co, aes(x=co_phase_rs[1:N], y=co_xdiff_rs[1:N])) +
  #ylim(-4,NA) +
  ylim(-14,NA) +  
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +   
  coord_polar(start =-1.45, theta='x', clip='off') +
  scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
p1_co
imageFile <- file.path(imageDirectory,"x-step_vs_phase_constant_polar.png")
ggsave(imageFile)



lm_xsm_sin_cos1 <- lm(sm_xdiff_mean[1,] ~ cos(sm_phase_rollmean)[1,] + sin(sm_phase_rollmean)[1,])
lm_xsm_linear <- lm(sm_xdiff_mean[1,] ~ sm_phase_rollmean[1,])
sm_xdiff_rs <- array_reshape(sm_xdiff_mean[1:1000,], dim=c(78*1000, 1)) # Is this correct?
sm_phase_rs <- array_reshape(sm_phase_rollmean[1:1000,], dim=c(78*1000, 1))
lm_xsmrs_sin_cos <- lm(sm_xdiff_rs ~ cos(sm_phase_rs) + sin(sm_phase_rs))
lm_xsmrs_linear <- lm(sm_xdiff_rs ~ sm_phase_rs)
summary(lm_xsmrs_sin_cos) # R-squared= 0.087, p < 2.2e-16
summary(lm_xsmrs_linear) # R-squared =2.3e-5, p=0.096
co_coefs_rs_linear = lm_xsmrs_linear$coefficients
anova(lm_xsmrs_sin_cos, lm_xsmrs_linear) # Sig. difference?!

#plot(sm_phase_rs, sm_xdiff_rs)
plot(sm_phase_rs[1:77,], sm_xdiff_rs[1:77,])
#plot(abs(sm_phase_rs), sm_xdiff_rs)
plot(abs(sm_phase_rs[1:77,]), sm_xdiff_rs[1:77,])
circlin.cor(sm_phase_rs, sm_xdiff_rs)

df_sm_xdiff_wide <- as.data.frame(sm_xdiff_mean[1:3,])

library(sandwich)
library(lmtest)
coeftest(lm_xsm_sin_cos1, vcov = vcovHC(lm_xsm_sin_cos1, type = "HC0"))


df_sm <- as.data.frame(cbind(sm_xdiff_rs[1:N], sm_phase_rs[1:N]))

# gp_dots_sm <- ggplot(data = df_sm, aes(x=sm_phase_rs[1:N], y=sm_xdiff_rs[1:N])) +
#   geom_point(alpha=0.4) +
#   geom_smooth() +
#   ylab("x-step") + xlab('Phase' ) +
#   ggtitle('Speed-modulated movement') +
#   theme(plot.title=element_text(hjust=0.5) ,
#         axis.text.x = element_text(face="bold", size=14),
#         axis.text.y = element_text(face="bold", size=14),
#         title=element_text(face = "bold", size = 16)) + # ,legend.position = 'none'
#   theme(legend.text = element_text(size = 20)) 
# plot(gp_dots_sm)
# imageFile <- file.path(imageDirectory,"x-step_vs_phase_speed-modulated.png")
# ggsave(imageFile)

p1_sm <- ggplot(data = df_sm, aes(x=sm_phase_rs[1:N], y=sm_xdiff_rs[1:N])) +
  #ylim(-4,NA) +
  ylim(-14,6) +  
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +   
  coord_polar(start =-1.45, theta='x', clip='off') +
  scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
p1_sm
imageFile <- file.path(imageDirectory,"x-step_vs_phase_speed-modulated_polar.png")
ggsave(imageFile)


lm_xaltrs_sin_cos <- lm(alt_xdiff_rs ~ cos(alt_phase_rs) + sin(alt_phase_rs))

lm_xaltrs_linear <- lm(alt_xdiff_rs ~ alt_phase_rs)
lm_xaltrs_const <- lm(alt_xdiff_rs ~ 1)

lm_xalt_sin_cos1 <- lm(alt_xdiff_mean[1,] ~ cos(alt_phase_rollmean)[1,] + sin(alt_phase_rollmean)[1,])
lm_xalt_linear <- lm(alt_xdiff_mean[1,] ~ alt_phase_rollmean[1,])

coefs_altrs <- lm_xaltrs_sin_cos$coefficients
print(coefs_altrs)
summary(lm_xaltrs_sin_cos) # Very simple solution!!!
modulation_altrs <- as.numeric(sqrt( coefs_altrs[2]**2 + coefs_altrs[3]**2 ) / abs(coefs_altrs[1]))
print(modulation_altrs)

# Re-sampling approach to characterize the distribution for true alternating movement:
coefs_alt <- array(data = NA, dim=c(3,1000))
modulation_alt <- array(data = NA, dim=1000)
for(i_rep in 1:1000){
  lm_xalt_sin_cos <- lm(alt_xdiff_mean[i_rep,] ~ cos(alt_phase_rollmean[i_rep,]) + sin(alt_phase_rollmean[i_rep,]))  
  coefs_alt[, i_rep] <- lm_xalt_sin_cos$coefficients    
  modulation_alt[i_rep] <- as.numeric(sqrt( coefs_alt[2, i_rep]**2 + coefs_alt[3, i_rep]**2 ) / abs(coefs_alt[1, i_rep]))    
}
print(quantile(modulation_alt, 0.95))
print(max(modulation_alt))
hist(modulation_alt, 50)

lm_xcors_sin_cos <- lm(co_xdiff_rs ~ cos(co_phase_rs) + sin(co_phase_rs))
coefs_cors <- lm_xcors_sin_cos$coefficients
print(coefs_cors)
summary(lm_xcors_sin_cos) # Very simple solution!!!
modulation_cors <- as.numeric(sqrt( coefs_cors[2]**2 + coefs_cors[3]**2 ) / abs(coefs_cors[1]))
print(modulation_cors)


#plot(co_phase_rs, co_xdiff_rs)
#plot(co_phase_rs[1:200], co_xdiff_rs[1:200])
pred_lm_co <- coefs_cors[1] + coefs_cors[2]*cos(co_phase_rs) + coefs_cors[3]*sin(co_phase_rs)
pred_lm_co_sortedphases <- sort(co_phase_rs[1:200], index.return=TRUE)
#lines(c(-pi,pi), c(coefs_cors[1], coefs_cors[1]))
#lines(pred_lm_co_sortedphases$x, pred_lm_co[pred_lm_co_sortedphases$ix])

#plot(abs(co_phase_rs[1:200]), co_xdiff_rs[1:200])
#lines(c(0,pi), c(coefs_cors[1], coefs_cors[1]))
pred_lm_co_sortedabsphases <- sort(abs(co_phase_rs[1:200]), index.return=TRUE)
lines(abs(pred_lm_co_sortedabsphases$x), coefs_cors[1] + coefs_cors[2]*cos(pred_lm_co_sortedabsphases$x) + coefs_cors[3]*sin(pred_lm_co_sortedabsphases$x))

# Re-sampling approach to derive limits for coefficients for true constant movement:
coefs_co <- array(data = NA, dim=c(3,1000))
modulation_co <- array(data = NA, dim=1000)
for(i_rep in 1:1000){
    lm_xco_sin_cos <- lm(co_xdiff_mean[i_rep,] ~ cos(co_phase_rollmean[i_rep,]) + sin(co_phase_rollmean[i_rep,]))  
    coefs_co[, i_rep] <- lm_xco_sin_cos$coefficients    
    modulation_co[i_rep] <- as.numeric(sqrt( coefs_co[2, i_rep]**2 + coefs_co[3, i_rep]**2 ) / abs(coefs_co[1, i_rep]))    
}
print(quantile(modulation_co, 0.95))
print(max(modulation_co))
hist(modulation_co, 50)


lm_xsmrs_sin_cos <- lm(sm_xdiff_rs ~ cos(sm_phase_rs) + sin(sm_phase_rs))
coefs_smrs <- lm_xsmrs_sin_cos$coefficients
print(coefs_smrs)
summary(lm_xsmrs_sin_cos) # Very simple solution!!!
modulation_smrs <- as.numeric(sqrt( coefs_smrs[2]**2 + coefs_smrs[3]**2 ) / abs(coefs_smrs[1]))
print(modulation_smrs)

# Re-sampling approach to characterize the distribution for true speed-modulated movement:
coefs_sm <- array(data = NA, dim=c(3,1000))
modulation_sm <- array(data = NA, dim=1000)
for(i_rep in 1:1000){
  lm_xsm_sin_cos <- lm(sm_xdiff_mean[i_rep,] ~ cos(sm_phase_rollmean[i_rep,]) + sin(sm_phase_rollmean[i_rep,]))  
  coefs_sm[, i_rep] <- lm_xsm_sin_cos$coefficients    
  modulation_sm[i_rep] <- as.numeric(sqrt( coefs_sm[2, i_rep]**2 + coefs_sm[3, i_rep]**2 ) / abs(coefs_sm[1, i_rep]))    
}
print(quantile(modulation_sm, 0.95))
print(quantile(modulation_sm, 0.05))
print(max(modulation_sm))
hist(modulation_sm, 50)








df_alt_phase_spikes <- data.frame(phase=alt_phase_rollmean[1,],
                                  spikes=alt_totspikes_rollmean[1,])

lm_alt_spikes_phase <- lm(spikes ~ cos(phase) + sin(phase), data=df_alt_phase_spikes) # Simple solution!!!
summary(lm_alt_spikes_phase) # Estimates are unbiased, statistics are biased
#library(performance)
#check_model(lm_alt_spikes_phase)



# Test with welch_anova_test:
#perform binning with specific number of bins
N <- 1000
df_co_bins <- as.data.frame(cbind(co_xdiff_rs[1:N], co_ydiff_rs[1:N], co_phase_rs[1:N], co_totspikes_rollmean_rs[1:N], co_xydiff_mean_rs[1:N]))
df_sm_bins <- as.data.frame(cbind(sm_xdiff_rs[1:N], sm_ydiff_rs[1:N], sm_phase_rs[1:N], sm_totspikes_rollmean_rs[1:N], sm_xydiff_mean_rs[1:N]))
df_alt_bins <- as.data.frame(cbind(alt_xdiff_rs[1:N], alt_ydiff_rs[1:N],alt_phase_rs[1:N], alt_totspikes_rollmean_rs[1:N], alt_xydiff_mean_rs[1:N]))
library(data.table)
setnames(df_co_bins, old=c('V1', 'V2', 'V3', 'V4', 'V5'), new=c('xdiff', 'ydiff', 'phase', 'spikes', 'xydiff'), skip_absent=TRUE)
setnames(df_sm_bins, old=c('V1', 'V2', 'V3', 'V4', 'V5'), new=c('xdiff', 'ydiff', 'phase', 'spikes', 'xydiff'), skip_absent=TRUE)
setnames(df_alt_bins, old=c('V1', 'V2', 'V3', 'V4', 'V5'), new=c('xdiff', 'ydiff', 'phase', 'spikes', 'xydiff'), skip_absent=TRUE)




library(dplyr)
n_bins = 16 # 4
df_co_bins <- df_co_bins %>% mutate(phase_bin = ntile(phase, n=n_bins)) # n=4
df_sm_bins <- df_sm_bins %>% mutate(phase_bin = ntile(phase, n=n_bins))
df_alt_bins <- df_alt_bins %>% mutate(phase_bin = ntile(phase, n=n_bins))

aggregate(df_co_bins$xdiff, list(df_co_bins$phase_bin), FUN=mean, na.rm=TRUE)
aggregate(df_co_bins$ydiff, list(df_co_bins$phase_bin), FUN=mean, na.rm=TRUE)
aggregate(df_co_bins$spikes, list(df_co_bins$phase_bin), FUN=mean, na.rm=TRUE)
aggregate(df_co_bins$phase, list(df_co_bins$phase_bin), FUN=mean, na.rm=TRUE)

df_co_bins$xdiff_binmean <- NULL
df_co_bins$ydiff_binmean <- NULL
df_co_bins$phase_binmean <- NULL
df_co_bins$spikes_binmean <- NULL
df_co_bins$error_term_meanspikes <- NULL

df_sm_bins$xdiff_binmean <- NULL
df_sm_bins$ydiff_binmean <- NULL
df_sm_bins$phase_binmean <- NULL
df_sm_bins$spikes_binmean <- NULL
df_sm_bins$error_term_meanspikes <- NULL

df_alt_bins$xdiff_binmean <- NULL
df_alt_bins$ydiff_binmean <- NULL
df_alt_bins$phase_binmean <- NULL
df_alt_bins$spikes_binmean <- NULL
df_alt_bins$error_term_meanspikes <- NULL
for (i_bin in 1 : n_bins ){ 
    df_co_bins$xdiff_binmean[df_co_bins$phase_bin == i_bin] <- mean(df_co_bins$xdiff[df_co_bins$phase_bin == i_bin])
    df_co_bins$ydiff_binmean[df_co_bins$phase_bin == i_bin] <- mean(df_co_bins$ydiff[df_co_bins$phase_bin == i_bin])
    df_co_bins$phase_binmean[df_co_bins$phase_bin == i_bin] <- mean(df_co_bins$phase[df_co_bins$phase_bin == i_bin])
    df_co_bins$spikes_binmean[df_co_bins$phase_bin == i_bin] <- mean(df_co_bins$spikes[df_co_bins$phase_bin == i_bin])
    df_co_bins$error_term_meanspikes[df_co_bins$phase_bin == i_bin] <- 15 * sqrt(pi/2 * 1/df_co_bins$spikes_binmean[df_co_bins$phase_bin == i_bin])
    
    df_sm_bins$xdiff_binmean[df_sm_bins$phase_bin == i_bin] <- mean(df_sm_bins$xdiff[df_sm_bins$phase_bin == i_bin])
    df_sm_bins$ydiff_binmean[df_sm_bins$phase_bin == i_bin] <- mean(df_sm_bins$ydiff[df_sm_bins$phase_bin == i_bin])
    df_sm_bins$phase_binmean[df_sm_bins$phase_bin == i_bin] <- mean(df_sm_bins$phase[df_sm_bins$phase_bin == i_bin])
    df_sm_bins$spikes_binmean[df_sm_bins$phase_bin == i_bin] <- mean(df_sm_bins$spikes[df_sm_bins$phase_bin == i_bin])
    df_sm_bins$error_term_meanspikes[df_sm_bins$phase_bin == i_bin] <- 15 * sqrt(pi/2 * 1/df_sm_bins$spikes_binmean[df_sm_bins$phase_bin == i_bin])

    df_alt_bins$xdiff_binmean[df_alt_bins$phase_bin == i_bin] <- mean(df_alt_bins$xdiff[df_alt_bins$phase_bin == i_bin])
    df_alt_bins$ydiff_binmean[df_alt_bins$phase_bin == i_bin] <- mean(df_alt_bins$ydiff[df_alt_bins$phase_bin == i_bin])
    df_alt_bins$phase_binmean[df_alt_bins$phase_bin == i_bin] <- mean(df_alt_bins$phase[df_alt_bins$phase_bin == i_bin])
    df_alt_bins$spikes_binmean[df_alt_bins$phase_bin == i_bin] <- mean(df_alt_bins$spikes[df_alt_bins$phase_bin == i_bin])
    df_alt_bins$error_term_meanspikes[df_alt_bins$phase_bin == i_bin] <- 15 * sqrt(pi/2 * 1/df_alt_bins$spikes_binmean[df_alt_bins$phase_bin == i_bin])
}
aggregate(df_co_bins$xdiff_binmean, list(df_co_bins$phase_bin), FUN=mean, na.rm=TRUE)
aggregate(df_co_bins$spikes_binmean, list(df_co_bins$phase_bin), FUN=mean, na.rm=TRUE)
aggregate(df_co_bins$error_term_meanspikes, list(df_co_bins$phase_bin), FUN=mean, na.rm=TRUE)

p_co_bins_xdiff <- ggplot(data = df_co_bins, aes(x=phase_bin, y=xdiff)) +
  #ylim(-4,NA) +
  #ylim(-14,6) +  
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +   
  #geom_bar(position=position_dodge(), stat='identity') + 
  stat_summary(fun = mean, geom = "bar", position="dodge") +  
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +     
  #geom_boxplot() +   
  geom_hline(yintercept=mean(co_xdiff_true_rs), linetype="solid", color = "red", linewidth=2) +        
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Constant movement')
  #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
p_co_bins_xdiff

p_co_bins_xydiff <- ggplot(data = df_co_bins, aes(x=phase_bin, y=sqrt(xdiff^2 + ydiff^2))) +
  #ylim(-4,NA) +
  #ylim(-14,6) +  
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +   
  stat_summary(fun = mean, geom = "bar", position="dodge") +  
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +     
  geom_hline(yintercept=mean(co_xydiff_true_rs), linetype="solid", color = "red", linewidth=2) +      
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Constant movement')  
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
p_co_bins_xydiff

p_sm_bins_xdiff <- ggplot(data = df_sm_bins, aes(x=phase_bin, y=xdiff)) +
  #ylim(-4,NA) +
  #ylim(-14,6) +  
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +   
  stat_summary(fun = mean, geom = "bar", position="dodge") +  
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +       
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Step-like movement')  
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
p_sm_bins_xdiff

p_alt_bins_xdiff <- ggplot(data = df_alt_bins, aes(x=phase_bin, y=xdiff)) +
  #ylim(-4,NA) +
  #ylim(-14,6) +  
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +   
  stat_summary(fun = mean, geom = "bar", position="dodge") +  
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +       
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Alternating movement')  
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
p_alt_bins_xdiff

p_sm_bins_xydiff <- ggplot(data = df_sm_bins, aes(x=phase_bin, y=sqrt(xdiff^2 + ydiff^2))) +
  #ylim(-4,NA) +
  #ylim(-14,6) +  
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +   
  stat_summary(fun = mean, geom = "bar", position="dodge") +  
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +       
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Step-like movement')    
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
p_sm_bins_xydiff

p_alt_bins_xydiff <- ggplot(data = df_alt_bins, aes(x=phase_bin, y=sqrt(xdiff^2 + ydiff^2))) +
  #ylim(-4,NA) +
  #ylim(-14,6) +  
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +   
  stat_summary(fun = mean, geom = "bar", position="dodge") +  
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +       
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Alternating movement')    
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
p_alt_bins_xydiff

p_sm_bins_spikes_binmean_polar <- ggplot(data = df_sm_bins, aes(x=phase_bin, y=spikes_binmean)) +
  #ylim(-4,NA) +
  ylim(0,100) +
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +
  #stat_summary(fun = mean, geom = "bar", position="dodge") +  
  #stat_summary(fun.data = mean_cl_boot, geom = "errorbar", position = position_dodge(width = 0.90), width = 0.2, color='black') +       
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Step-like movement')    
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
p_sm_bins_spikes_binmean_polar

p_sm_bins_error_meanspikes_polar <- ggplot(data = df_sm_bins, aes(x=phase_binmean, y=error_term_meanspikes)) +
  ylim(0,NA) +
  #ylim(-10,10) +
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Step-like movement')  
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
p_sm_bins_error_meanspikes_polar
  
# 
# p_co_bins_xdiff_binmean <- ggplot(data = df_co_bins, aes(x=phase_bin, y=xdiff_binmean)) +
#   #ylim(-4,NA) +
#   ylim(-10,10) +  
#   geom_point(color='purple',stat='identity', alpha=0.4)+
#   geom_smooth() +   
#   coord_polar(start =-1.45, theta='x', clip='off') #+
# #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
# p_co_bins_xdiff_binmean
# 
# p_co_bins_xydiff_binmean_polar <- ggplot(data = df_co_bins, aes(x=phase_bin, y=sqrt(xdiff_binmean^2 + xdiff_binmean^2))) +
#   #ylim(-4,NA) +
#   ylim(-10,10) +  
#   geom_point(color='purple',stat='identity', alpha=0.4)+
#   geom_smooth() +   
#   coord_polar(start =-1.45, theta='x', clip='off') #+
# #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
# p_co_bins_xydiff_binmean_polar
# 
p_co_bins_spikes_binmean_polar <- ggplot(data = df_co_bins, aes(x=phase_bin, y=spikes_binmean)) +
  #ylim(-4,NA) +
  ylim(0,100) +
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Constant movement')  
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
p_co_bins_spikes_binmean_polar
# 
p_co_bins_xydiff_binmean_cartes <- ggplot(data = df_co_bins, aes(x=abs(phase_binmean), y=sqrt(xdiff_binmean^2 + xdiff_binmean^2))) +
  #ylim(-4,NA) +
  ylim(-10,10) +
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth(method='lm') #+
   coord_polar(start =-1.45, theta='x', clip='off') #+
  scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
p_co_bins_xydiff_binmean_cartes
# 
# p_co_bins_spikes_binmean_cartes <- ggplot(data = df_co_bins, aes(x=abs(phase_binmean), y=spikes_binmean)) +
#   #ylim(-4,NA) +
#   #ylim(0,100) +  
#   geom_point(color='purple',stat='identity', alpha=0.4)+
#   geom_smooth() #+   
#   #coord_polar(start =-1.45, theta='x', clip='off') #+
# #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
# p_co_bins_spikes_binmean_cartes
# 
p_co_bins_error_meanspikes_polar <- ggplot(data = df_co_bins, aes(x=phase_binmean, y=error_term_meanspikes)) +
  ylim(0,NA) +
  #ylim(-10,10) +
  geom_point(color='purple',stat='identity', alpha=0.4)+
  geom_smooth() +
  coord_polar(start =-1.45, theta='x', clip='off') +
  ggtitle('Constant movement')
#scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # +
p_co_bins_error_meanspikes_polar
# 
# p_co_bins_error_meanspikes_cartes <- ggplot(data = df_co_bins, aes(x=abs(phase_binmean), y=error_term_meanspikes)) +
#   #ylim(-4,NA) +
#   #ylim(-10,10) +  
#   geom_point(color='purple',stat='identity', alpha=0.4)+
#   geom_smooth() #+   
# #coord_polar(start =-1.45, theta='x', clip='off') #+
# #scale_x_continuous(name="angle", limits=c(-1.45, 1.45)) # + 
# p_co_bins_error_meanspikes_cartes

library(patchwork)
p_sm_bins_xydiff + p_co_bins_xydiff + p_alt_bins_xydiff + p_sm_bins_xdiff  + p_co_bins_xdiff  + p_alt_bins_xdiff
imageFile <- file.path(imageDirectory,"Overview_polar_xdiff_and_xydiff-vs-phase.png")
ggsave(imageFile)

p_sm_bins_spikes_binmean_polar + p_co_bins_spikes_binmean_polar + p_sm_bins_error_meanspikes_polar + p_co_bins_error_meanspikes_polar
imageFile <- file.path(imageDirectory,"Overview_polar_spikes_and_error-vs-phase.png")
ggsave(imageFile)

# For the article:
N <- 1000

# Combine the four correlation plots into one figure
combined_plot <- (gp_dots_xystep_sm + gp_dots_xy_co_phase) / 
  (gp_dots_sm + gp_dots_co) +
  plot_annotation(tag_levels = list(c('a', 'b', 'c', 'd'))) &
  theme(plot.tag = element_text(face = "bold", size = 28))

print(combined_plot)

imageFile <- file.path(imageDirectory, "fig_compare_correl.png")
ggsave(imageFile, plot = combined_plot, width = 14, height = 10, dpi = 300)

