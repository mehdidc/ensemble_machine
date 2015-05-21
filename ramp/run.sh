#!/bin/sh

#for lambda in 0.001 0.01 0.1 0.2 0.25 0.30 0.4 0.5 0.6 0.7 0.75 0.8 0.9 0.99 0.999 0.9999; do
for lambda in 0.9 0.91 0.92 0.95 0.99 0.995 0.999 0.9995 0.9999; do
    cat ramp.py | sed -e "s/lambda_ = 0.1/lambda_=$lambda/g"|python
done
