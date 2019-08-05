#!/bin/bash

set -ex

# Script to reproduce results

for ((i=0;i<10;i+=1))
do
    # python main.py \
    # --policy_name "TD3" \
    # --env_name "HalfCheetah-v1" \
    # --seed $i \
    # --start_timesteps 10000

    # python main.py \
    # --policy_name "TD3" \
    # --env_name "Hopper-v1" \
    # --seed $i \
    # --start_timesteps 1000

    python main.py \
    --policy_name "TD3" \
    --env_name "BipedalWalker-v2" \
    --seed $i \
    --start_timesteps 0

    exit 0

    # python main.py \
    # --policy_name "TD3" \
    # --env_name "Ant-v1" \
    # --seed $i \
    # --start_timesteps 10000

    # python main.py \
    # --policy_name "TD3" \
    # --env_name "InvertedPendulum-v1" \
    # --seed $i \
    # --start_timesteps 1000

    # python main.py \
    # --policy_name "TD3" \
    # --env_name "InvertedDoublePendulum-v1" \
    # --seed $i \
    # --start_timesteps 1000

    # python main.py \
    # --policy_name "TD3" \
    # --env_name "Reacher-v1" \
    # --seed $i \
    # --start_timesteps 1000
done
