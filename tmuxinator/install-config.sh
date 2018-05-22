#!/usr/bin/env bash

mkdir -p ~/.config/tmuxinator
ln -s ${PWD}/lr-dist-example.yml ~/.config/tmuxinator/lr-dist-example.yml
echo "Installed tmuxinator config successfully"