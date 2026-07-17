library(xtable)
library(ggplot2)
library(reshape2)
library(tidyr)
library(ggpubr)
library(dplyr)
library(reshape2)
library(ggnewscale)
library(gifski)     # To animate
library(gapminder)  # To animate
library(gganimate)  # To animate
options(xtable.floating = TRUE)
options(xtable.timestamp = "")

rm(list=ls())

load((paste0("subset_sim_complete.RData")))
plot_size <- c(8, 5) / 1

# Process data.
dataframe.print <- results[[1]] #all.evals.table.1
dataframe.print <- dataframe.print %>% filter((covariance == -0.9) | (covariance == 0)| (covariance == 0.9))
dataframe.print <- select(dataframe.print, -c(Var1, SMA))

# Plot of trajectories.
trajectories.df <- results[[2]]

trajectories.df$het_A <- factor(trajectories.df$het_A)
levels(trajectories.df$het_A) <- c("homo. A", "het. A")
trajectories.df$radius <- factor(trajectories.df$radius)
trajectories.df$dist <- factor(trajectories.df$dist)
trajectories.df$type_char <- 'Y-shaped'
trajectories.df$type_char[trajectories.df$iSimulationType == 3] <- 'Flipped-Y'
trajectories.df$simulation <- factor(paste0("N = ", trajectories.df$N, "; T = ", trajectories.df$T))

trajectories_subset <- trajectories.df
plot_dens <- ggplot(trajectories_subset,
                    aes(x=x, y=y, color=factor(cluster_idx))) +
  geom_density2d(data = subset(trajectories_subset, true_or_est==0))+
  geom_path(data=subset(trajectories_subset, (true_or_est == 1) & (sim_run == "V1") & (dGamma == 0.25)),
            color='black', aes(group=cluster_idx)) + xlim(c(-3, 3)) + ylim(c(-3, 3)) +
  facet_grid(radius+iSimulationType~covariance) + coord_fixed() + scale_color_discrete(guide='none')
ggsave(paste0('Results/trajectories_of_means_densities.png'),
       plot=plot_dens, width = 10)

# Keep only the first 10 simulation runs for the scatter plots below.
trajectories_first_10 <- trajectories.df[trajectories.df$sim_run %in% paste0("V", 1:10),]
levels(trajectories_first_10$het_A) <- c("Homogeneous A", "Heterogeneous A")
trajectories_first_10$radius <- as.character(trajectories_first_10$radius)
trajectories_first_10$dist <- as.character(trajectories_first_10$dist)
plot_scatter <- ggplot(trajectories_first_10,
                    aes(x=x, y=y, color=factor(cluster_idx), alpha = 0.5)) +
  geom_point(data = subset(trajectories_first_10, true_or_est==0))+
  geom_path(data=subset(trajectories_first_10, (true_or_est == 1) & (sim_run == "V1") & (dGamma == 0.25)),
            color='black', aes(group=cluster_idx)) + xlim(c(-3, 3)) + ylim(c(-3, 3)) +
  facet_grid(radius+iSimulationType~covariance,
             labeller = label_bquote(rows = "radius ="~.(radius),
                                     cols = "Cov ="~.(covariance))) +
  coord_fixed() +
  scale_color_discrete(guide='none') + scale_alpha(guide='none') +
  theme_classic()
ggsave(paste0('Results/trajectories_of_means_scatter.png'),
       plot=plot_scatter, width = 10)

plot_scatter_small_weighted <- ggplot(trajectories_first_10,
                       aes(x=x, y=y, alpha=0.5, color = total_correct_class)) +
  geom_point(data = subset(trajectories_first_10, (true_or_est==0)))+
  geom_path(data=subset(trajectories_first_10, (true_or_est == 1) & (sim_run == "V1") &
                          (het_A == 'Homogeneous A') & (dGamma == 0.25)),
            color='black', aes(group=cluster_idx)) +
  facet_grid(radius+type_char~covariance,
             labeller = label_bquote(rows = "Size ="~.(radius),
                                     cols = "Cov. ="~.(covariance))) +
  coord_fixed() +
  scale_alpha(guide='none') + scale_color_continuous(name="Correct classification") +
  theme_classic() + theme(legend.position = "bottom")

ggsave(paste0('Results/trajectories_of_means_scatter_small_weighted.png'),
       plot=plot_scatter_small_weighted, width = plot_size,
       height = plot_size)

plot_scatter_small_weighted_alpha <- ggplot(trajectories_first_10,
                                      aes(x=x, y=y,
                                          color = factor(cluster_idx),
                                          alpha = total_correct_class)) +
  geom_point(data = subset(trajectories_first_10, (true_or_est==0))) +
  geom_path(data=subset(trajectories_first_10, (true_or_est == 1) & (sim_run == "V1") &
                          (het_A == 'Homogeneous A') & (dGamma == 0.25)),
            color='black', aes(group=cluster_idx)) +
  facet_grid(radius+type_char~covariance,
             labeller = label_bquote(rows = "Size ="~.(radius),
                                     cols = "Cov. ="~.(covariance))) +
  coord_fixed() +
  scale_color_discrete(guide='none') + scale_alpha(name="Correct classification") +
  theme_classic() + theme(legend.position = "bottom")

ggsave(paste0('Results/trajectories_of_means_scatter_small_weighted_alpha.png'),
       plot=plot_scatter_small_weighted_alpha, width = plot_size,
       height = plot_size)

trajectories_paths <- trajectories.df[trajectories.df$sim_run %in% paste0("V", 1:10),]
plot_path <- ggplot(trajectories_paths, aes(x=x, y=y, linetype=factor(cluster_idx), color=factor(sim_run))) +
  geom_path(data = subset(trajectories_paths, true_or_est==0)) +
  facet_grid(radius+dist~covariance+het_A) + coord_fixed() +
  scale_color_discrete(guide='none') +  xlim(c(-3, 3)) + ylim(c(-3, 3)) +
  scale_linetype_discrete(guide='none')
ggsave(paste0('Results/trajectories_of_means_path.png'),
       plot=plot_path, width = 10)

plot_path_all <- ggplot(trajectories.df, aes(x=x, y=y, color=factor(cluster_idx),
                                             group=interaction(factor(cluster_idx),
                                                               factor(sim_run)))) +
  geom_path(data = subset(trajectories.df, true_or_est==0), alpha=0.25) +
  facet_grid(radius+dist~covariance+het_A) + coord_fixed() +
  scale_color_discrete(guide='none') + xlim(c(-5, 5)) + ylim(c(-5, 5)) +
  scale_linetype_discrete(guide='none') + scale_alpha(guide='none')
ggsave(paste0('Results/trajectories_of_means_path_all.png'),
       plot=plot_path_all, width = 10)

plot_path_all_small <- ggplot(trajectories.df, aes(x=x, y=y, color=factor(cluster_idx),
                                             group=interaction(factor(cluster_idx),
                                                               factor(sim_run)))) +
  geom_path(data = subset(trajectories.df, true_or_est==0), alpha=0.25) +
  facet_grid(radius+dist~covariance) + coord_fixed() +
  scale_color_discrete(guide='none') + xlim(c(-5, 5)) + ylim(c(-5, 5)) +
  scale_linetype_discrete(guide='none') + scale_alpha(guide='none')
ggsave(paste0('Results/trajectories_of_means_path_all_small.png'),
       plot=plot_path_all_small, width = 10)

# Plot of trajectories' means and quantiles.
quantile_source <- subset(trajectories.df, (true_or_est == 0) |
                 ((true_or_est == 1) & (sim_run == "V1")))
quantile_source <- quantile_source %>% select(-c(SMA, TV,
                           Orient_1, Orient_2,
                           Var1, Var2, smoothDGP, smoothEst, N, T,
                           simulation))
quantile_data <- quantile_source %>% group_by(het_A, radius, iSimulationType, covariance, true_or_est, cluster_idx, t) %>%
  summarize(x_m = mean(x), y_m = mean(y),
            quant = scales::percent(c(0.25, 0.5, 0.75)),
            x_q = quantile(x, c(0.25, 0.5, 0.75)),
            y_q = quantile(y, c(0.25, 0.5, 0.75)),
            quant_5 = scales::percent(c(0.05, 0.5, 0.95)),
            x_q_5 = quantile(x, c(0.05, 0.5, 0.95)),
            y_q_5 = quantile(y, c(0.05, 0.5, 0.95))) %>% ungroup()

quantile_data_simulated <- subset(quantile_data, (true_or_est == 0))
quantile_data_all <- subset(quantile_data)
plot_quantiles_all <- ggplot(quantile_data_simulated, aes(x = x_q, y = y_q, color = factor(quant),
                                       group = interaction(factor(cluster_idx),
                                                           factor(quant)))) +
  geom_path() +
  geom_path(aes(x = x_m, y=y_m,
                group = interaction(factor(cluster_idx), factor(true_or_est)),
                linetype=factor(true_or_est)), data=quantile_data_all,
            color = 'black') +
  xlab('x') + ylab('y') + coord_fixed() + facet_grid(radius+iSimulationType~covariance) +
  scale_color_discrete(name='Quantile') +
  scale_linetype_discrete(name='', labels=c('Mean', 'DGP'))
ggsave(paste0('Results/trajectories_of_means_quantiles_all.png'),
       plot=plot_quantiles_all, width = 10)

# Misclassification rate over time.
misclass_per_time <- trajectories.df %>% filter(cluster_idx == 1, true_or_est == 0, exit_status ==0)
misclass_per_time$radius <- as.numeric(as.character(misclass_per_time$radius))
misclass_per_time_agg <- misclass_per_time %>% group_by(radius, type_char, covariance, t)
misclass_per_time <- misclass_per_time %>% group_by(radius, type_char, covariance, dGamma, t)
misclass_per_time_agg <- misclass_per_time_agg %>%
  summarize(MSE_m = mean(total_MSE_over_time), class_m = mean(total_correct_class),
          quant = scales::percent(c(0.25, 0.5, 0.75)),
          MSE_q = quantile(total_MSE_over_time, c(0.25, 0.5, 0.75)),
          class_q = quantile(total_correct_class, c(0.25, 0.5, 0.75)),
          quant_5 = scales::percent(c(0.05, 0.5, 0.95)),
          MSE_q_5 = quantile(total_MSE_over_time, c(0.05, 0.5, 0.95)),
          class_q_5 = quantile(total_correct_class, c(0.05, 0.5, 0.95))) %>% ungroup()
misclass_per_time <- misclass_per_time %>%
  summarize(MSE_m = mean(total_MSE_over_time), class_m = mean(total_correct_class),
            quant = scales::percent(c(0.25, 0.5, 0.75)),
            MSE_q = quantile(total_MSE_over_time, c(0.25, 0.5, 0.75)),
            class_q = quantile(total_correct_class, c(0.25, 0.5, 0.75)),
            quant_5 = scales::percent(c(0.05, 0.5, 0.95)),
            MSE_q_5 = quantile(total_MSE_over_time, c(0.05, 0.5, 0.95)),
            class_q_5 = quantile(total_correct_class, c(0.05, 0.5, 0.95))) %>% ungroup()

misclass_per_time_agg_numeric <- misclass_per_time_agg
misclass_per_time_agg_numeric$quant <- as.numeric(substr(misclass_per_time_agg_numeric$quant, 1, 2))/100
misclass_per_time_agg_bands <-
  reshape(data.frame(misclass_per_time_agg_numeric), direction="wide",
          v.names=c("MSE_q", "MSE_m", "class_q", "class_m"),
          timevar=c("quant"),
          idvar=c("radius", "type_char", "covariance", "t"),
          drop=c("quant_5", "MSE_q_5", "class_q_5" ))

plot_mse_small <- ggplot(misclass_per_time_agg_bands, aes(y=MSE_q.0.5, x=t)) + geom_line() +
  facet_grid(radius+type_char~covariance,
             labeller = label_bquote(rows = "Size ="~.(radius)~";"~.(type_char),
                                     cols = "Cov. ="~.(covariance))) +
  xlab('Time') + ylab('Squared distance from the true mean') +
  geom_ribbon(aes(ymin=MSE_q.0.25,ymax=MSE_q.0.75), fill="black", alpha=0.3) +
  ylim(0, 1)  + theme_classic() + theme(legend.position = 'bottom') +
  grids(linetype = "solid")

ggsave(paste0('Results/MSE_overe_time_small.png'),
       plot=plot_mse_small, width = plot_size,
       height = plot_size)

plot_class_small <- ggplot(misclass_per_time_agg_bands, aes(y=class_q.0.5, x=t)) + geom_line() +
  facet_grid(radius+type_char~covariance,
             labeller = label_bquote(rows = "Size ="~.(radius)~";"~.(type_char),
                                     cols = "Cov. ="~.(covariance))) +
  xlab('Time') + ylab('Correct classification') +
  geom_hline(yintercept=0.5, linetype="dashed") +
  geom_ribbon(aes(ymin=class_q.0.25,ymax=class_q.0.75), fill="black", alpha=0.3) +
  ylim(0.4, 1)  + theme_classic() +
  grids(linetype = "solid")

ggsave(paste0('Results/class_overe_time_small.png'),
       plot=plot_class_small, width = plot_size,
       height = plot_size)

plot_mse_big <- ggplot(misclass_per_time, aes(y=MSE_q, x=t, color = factor(quant))) + geom_line() +
  facet_grid(radius+type_char~covariance+dGamma) +
  xlab('Time') + ylab('Squared distance from the true mean') + scale_color_discrete(name="Quantile")
ggsave(paste0('Results/MSE_overe_time_big.png'),
       plot=plot_mse_big, width = plot_size,
       height = plot_size)

plot_class_big <- ggplot(misclass_per_time, aes(y=class_q, x=t, color = factor(quant))) + geom_line() +
  facet_grid(radius+type_char~covariance+dGamma) +
  xlab('Time') + ylab('Correct classification') + scale_color_discrete(name="Quantile")
ggsave(paste0('Results/class_overe_time_big.png'),
       plot=plot_class_big, width = plot_size,
       height = plot_size)

# Summary table.
table_title <- "Results for N = 100, T = 40."

print_df <- dataframe.print %>% filter(radius==2) %>%
  group_by(N, T, radius, dist, covariance, dGamma, het_A) %>%
  dplyr::summarise_each(mean)

print_df$het_A <- factor(print_df$het_A)
print_df <- print_df[c(3, 5, 6, 8, 17:length(names(print_df)))]

names(print_df) <- c("Radius", "Cov.", "$\\gamma$", "Sim.",
                     "$\\hat \\gamma$", "Class.", "MSE", "LL",
                     "Class. (Ward)", "MSE (Ward)")

digits_for_table <- c(0, 0, 1, 1, 2, 0, rep(3, ncol(print_df)-5))
df.xtab <- xtable(print_df,
                  digits=digits_for_table,
                  caption=table_title)

table_to_latex <- print(df.xtab, include.rownames=FALSE,
                        sanitize.text.function=function(x){x},
                        file=paste0(
                                    "Results/focus_table.tex"),
                        caption.placement = 'top')
