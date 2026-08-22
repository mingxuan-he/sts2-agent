#!/bin/sh
# Inner session entrypoint. The supervisor runs this in a loop; each run is
# one session (one fresh context). This file is yours — restructure it if a
# different session shape serves you better.
cd /pod/harness && exec node main.js
