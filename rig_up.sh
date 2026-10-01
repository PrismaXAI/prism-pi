#!/bin/bash
# YAM rig start-up after a reboot / adapter replug: CAN adapters -> can0 (RIGHT) / can1 (LEFT) at 1 Mbit, then the web UI.
# Idempotent: safe to run again. Moves no arms, deletes nothing. Needs sudo for `ip link`. Usage: ./rig_up.sh
#   PRISM_CKPT=<checkpoint dir> ./rig_up.sh   also pre-loads the policy server (otherwise the UI starts it on the first session)
#   PYTHON=<interpreter>                      default .venv/bin/python;  YAM_UI_PORT=<port>  default 8080
cd "$(dirname "$0")"
PYTHON=${PYTHON:-.venv/bin/python}
PORT=${YAM_UI_PORT:-8080}
declare -a TMP
# CAN adapters are identified by their USB serial and named by what they DRIVE (config/rig_mapping.json): code convention
# LEFT arm = can1, RIGHT arm = can0. Two passes through temporary names so a swap never collides.
declare -A WANT
while read -r serial name; do WANT[$serial]=$name; done < <($PYTHON -m prism_pi.rig.mapping can)
[ "${#WANT[@]}" -gt 0 ] || echo "!! config/rig_mapping.json has no can_by_serial entries (copy config/rig_mapping.example.json and fill it in)"
i=0
for dev in /sys/class/net/can*; do
  [ -e "$dev" ] || continue
  cur=$(basename "$dev"); serial=$(udevadm info -a "$dev" 2>/dev/null | grep -m1 'ATTRS{serial}' | sed 's/.*=="\(.*\)"/\1/')
  if [ -n "${WANT[$serial]}" ]; then
    sudo ip link set "$cur" down; sudo ip link set "$cur" name "cantmp$i"; TMP[$i]="${WANT[$serial]}"; i=$((i+1))
  else
    echo "!! $cur has unknown adapter serial '$serial' (not in config/rig_mapping.json)"
  fi
done
for j in $(seq 0 $((i-1))); do
  sudo ip link set "cantmp$j" name "${TMP[$j]}"; sudo ip link set "${TMP[$j]}" up type can bitrate 1000000
done
[ "$i" -eq 2 ] || echo "!! expected 2 CAN adapters, found $i (ip -br link | grep can)"
sleep 1; ip -br link | grep -E "^can"
if [ -e ESTOP ]; then echo "!! e-stop latch is set (file ESTOP): press 'Reset latch' in the web UI before starting a session"; fi
if ! ss -ltn | grep -q ":$PORT "; then
  YAM_UI_PORT=$PORT setsid nohup $PYTHON -u -m prism_pi.webui >> webui.log 2>&1 < /dev/null &
  echo "web UI starting on http://$(hostname -I | awk '{print $1}'):$PORT"
else
  echo "web UI already running"
fi
if [ -n "$PRISM_CKPT" ]; then
  if ! $PYTHON -m prism_pi.policy.client info 2>/dev/null | grep -q "'ckpt'"; then
    setsid nohup env HF_HUB_OFFLINE=1 $PYTHON -u -m prism_pi.policy.server --ckpt "$PRISM_CKPT" >> policy_server.log 2>&1 < /dev/null &
    echo "policy server starting with $PRISM_CKPT (~1 min)"
  else
    echo "policy server already running"
  fi
fi
sleep 12
curl -s localhost:$PORT/status | $PYTHON -c "import json,sys;s=json.load(sys.stdin);print('UI:',s['session']['state'],'| cameras:',{k:('OK %d fps'%round(v['fps']) if v['ok'] else 'OFFLINE') for k,v in s['cams'].items()},'| CAN:',s['can'])" 2>/dev/null || echo "!! web UI not answering on :$PORT yet (see webui.log)"
