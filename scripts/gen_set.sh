#!/usr/bin/bash

# shellcheck disable=SC2045

if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="/wj-vol/mani_skill_assets/data"
fi

if [[ -f "$MS_ASSET_DIR/data/mshab_checkpoints" ]]; then
    CKPT_DIR="$MS_ASSET_DIR/data/mshab_checkpoints"
else
    CKPT_DIR="mshab_checkpoints"
fi

task="set_table"

for subtask in $(ls -1 "$CKPT_DIR/rl/$task"); do
    if [[ $subtask == "close" || $subtask == "open" ]]; then
        continue
    fi

    for obj_name in $(ls -1 "$CKPT_DIR/rl/$task/$subtask"); do
        if [[ $obj_name == "all" ]]; then
            python -m mshab.utils.gen.gen_data "$task" "$subtask" "$obj_name"
        fi
    done
done