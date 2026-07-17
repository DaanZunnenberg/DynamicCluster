library(xtable)
library(ggplot2)
library(reshape2)
library(tidyr)
library(dplyr)
library(foreach)
library(mlbench)
library(data.table)  # fread function
options(xtable.floating = TRUE)
options(xtable.timestamp = "")

rm(list=ls())

# Study configuration, must match the settings used to produce the
# simulation results CSV read in below.
simulation_type <- 3
n_units_per_cluster <- 100
n_time_steps <- 40
mean_specific_smoothing <- 0
circle_midpoint_distance = 2
circle_radius = 2
n_clusters <- 2
n_features <- 2
gamma <- 0.25
n_repetitions <- 1
semi_major_axis <- 2
simulation_variance_1 <- 1
simulation_variance_2 <- 1
covariance <- 0.9
ellipse1_orientation <- 1
ellipse2_orientation <- 0
smoothing_dgp <- 0.1
smoothing_estimation <- 0.25
n_clusters_estimation <- 2
time_varying_probabilities <- 1  # 0 means constant transition matrix, 1 means time-varying transition matrix

simulation_identifiers <- c("iSimulationType","N", "T", "iD", "radius", "dist", "het_A",
                            "iP", "TV", "dGamma", "SMA", "Orient_1",
                            "Orient_2", "Var1", "Var2", 'covariance', 'smoothDGP', 'smoothEst')

simulation_params <-  c(simulation_type, n_units_per_cluster, n_time_steps, n_features,
                        circle_radius, circle_midpoint_distance,
                        mean_specific_smoothing, n_clusters_estimation, time_varying_probabilities,
                        gamma, semi_major_axis,
                        ellipse1_orientation, ellipse2_orientation, simulation_variance_1,
                        simulation_variance_2, covariance, smoothing_dgp, smoothing_estimation)

setting.here <- data.frame(rbind(simulation_params,simulation_params),stringsAsFactors = FALSE)[1,]
colnames(setting.here) <- simulation_identifiers

# As.df: saves estimates of A's
As.df <- as.data.frame(matrix(nrow=0, ncol=n_repetitions+1+length(simulation_identifiers)))

# trajectories.df: saves the trajectories of the estimated means for plotting.
# 2 cols for x and y, 1 for simulation run, 1 for simulation design, 1 for gamma,
# 1 for true/estimated, 1 for time, 1 for cluster 1/2.
trajectories.names <-  c(simulation_identifiers,
                         'true_or_est', 'cluster_idx',
                         't', 'x', 'y', 'total_MSE_over_time', 'total_correct_class', "exit_status")
trajectories.df <- as.data.frame(matrix(nrow=0, ncol=length(trajectories.names)))
names(trajectories.df) <- trajectories.names

# Number of means and unique covariance elements per cluster at time t.
n_params_per_cluster <- n_features+(n_features^2+n_features)/2
file.name <- paste("Sim_results_3_100_40_0.25_2_2_0_0_2_2_1_0_2_1_0_1_1_0.9_0.25_0.1", ".csv",sep="")

result.coll.1 <- fread(file.name)
# Print optimizer status.
print(paste0("File read. # of non-convergence: ", sum(result.coll.1[2,] !=0),
             ". File name: ", file.name))

if (mean_specific_smoothing == 1){
  A1s <- data.frame(result.coll.1[3:4,])
  A1s <- cbind(data.frame(a_index = c(1, 2), setting.here), A1s)
} else{
  A1s <- data.frame((result.coll.1[3,]))
  A1s <- cbind(data.frame(a_index = 3, setting.here), A1s)
}

# Our clustering method's results.
eval_failures <- result.coll.1[2,] == 4
eval_failures <- result.coll.1[2,] != 0
sims_with_nas <- lapply(lapply(result.coll.1, is.na), sum) > 0
total_to_skip <- eval_failures | sims_with_nas
print(paste0(" Our clustering; # of function eval. failures: ",
             sum(eval_failures), "; # total skipping due to NA: ", sum(sims_with_nas)))

evals.lss.all <- lapply(result.coll.1[,-..total_to_skip], eval.round, setting.here)

traj_fun_in_nr.simulations <- function(evals.lss, k){
  trajectories_list <- list(estimated.means.cluster1 = evals.lss$estimated.means.cluster1,
                            estimated.means.cluster2 = evals.lss$estimated.means.cluster2,
                            true.means.cluster1 = evals.lss$true.means.cluster1,
                            true.means.cluster2 = evals.lss$true.means.cluster2)
  trajectories.here <- do.call(rbind, lapply(trajectories_list, t))

  trajectories.df.this.run <- data.frame(c(setting.here),
                                         rep(0:1, each=setting.here$T*2),
                                         rep(rep(1:2, each=setting.here$T), 2),
                                         rep(1:setting.here$T, 4),
                                         trajectories.here[,1], trajectories.here[,2],
                                         rep(evals.lss$MSE.p_t, 4),
                                         rep(evals.lss$correct.class_per_time, 4),
                                         rep(evals.lss$exit_status, (setting.here$T)*4))
  names(trajectories.df.this.run) <- names(trajectories.df)

  if (setting.here$iP >2){
    n_of_extra_clusters <- setting.here$iP - 2
    extra_trajectories <- do.call(rbind, lapply(evals.lss$other_cluster_estimated_means, t))
    extra_clusters <- data.frame(c(setting.here),
                                 rep(0, setting.here$T*n_of_extra_clusters),
                                 rep(3:setting.here$iP, each=setting.here$T),
                                 rep(1:setting.here$T, n_of_extra_clusters),
                                 extra_trajectories,
                                 rep(evals.lss$MSE.p_t, n_of_extra_clusters),
                                 rep(evals.lss$correct.class_per_time, n_of_extra_clusters),
                                 rep(evals.lss$exit_status, (setting.here$T)*n_of_extra_clusters))
    names(extra_clusters) <- names(trajectories.df)
    trajectories.df.this.run <- rbind(trajectories.df.this.run, extra_clusters)
  }
  return(trajectories.df.this.run)
}

evals_fun_in_nr.simulations <- function(evals.lss){
  return(c(evals.lss$gamma.hat, evals.lss$correct.class,
                        evals.lss$average.MSE, evals.lss$likelihood))
}

tmp_trajectories <- mapply(traj_fun_in_nr.simulations,
                           evals.lss.all, SIMPLIFY=FALSE)

for (sim_i in 1:length(tmp_trajectories)){
  tmp_trajectories[[sim_i]]$sim_run <- gsub("Var", "Sim", names(tmp_trajectories)[sim_i])
}
trajectories_h <- bind_rows(tmp_trajectories)
trajectories_h$sim_run <- factor(trajectories_h$sim_run)

evals.table.1 <- t(bind_rows(lapply(evals.lss.all, evals_fun_in_nr.simulations)))

# Hierarchical clustering baseline's results.
print(paste0(" Hierarchical clustering; # of function eval. failures: ",
             sum(eval_failures), "; # total skipping due to NA: ", sum(sims_with_nas)))

evals.lss.all.ward <- lapply(result.coll.1[,-..total_to_skip], eval.hierarchical, setting.here)
evals_fun_in_nr.simulations.ward <- function(evals.lss){
  return(c(evals.lss$class.outcome, evals.lss$average.MSE))
}
evals.table.1.ward <- t(bind_rows(lapply(evals.lss.all.ward,
                                         evals_fun_in_nr.simulations.ward)))

all.evals.table.h <- data.frame(c(setting.here), 1:nrow(evals.table.1), evals.table.1, evals.table.1.ward)
names(all.evals.table.h)[(length(setting.here)+1):ncol(all.evals.table.h)] <-
  c('simulation', 'hat_gamma', 'class', 'MSE', 'LL', 'class_ward', 'MSE_ward')

dfs <- list(
  all.evals.table.h,
  trajectories_h,
  A1s)

combine_dfs <- function(df_list){
  res <- list()
  for(k in 1:ncol(df_list)){
    tmp <- list()
    for (i in 1:nrow(df_list)){
      tmp <- c(tmp, df_list[i, k])
    }
    res[[k]] <- bind_rows(tmp)
  }
  return(res)
}

results <- combine_dfs(rbind(dfs))

print("Saving")
save.image(paste0("subset_sim_complete.RData"))
