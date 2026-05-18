#!/usr/bin/env bash
# Control the tr-agent supervisord process.
# Usage: ./agent.sh {start|stop|restart|status|logs}

CONF="$(dirname "$0")/supervisord.conf"
CTL=".venv/bin/supervisorctl"
SD=".venv/bin/supervisord"

cd "$(dirname "$0")" || exit 1

case "${1:-status}" in
  start)
    if $CTL -c "$CONF" status tr-agent &>/dev/null; then
      echo "Already running. Use './agent.sh restart' to reload."
    else
      $SD -c "$CONF"
      echo "Started."
    fi
    ;;
  stop)
    $CTL -c "$CONF" stop tr-agent
    ;;
  restart)
    $CTL -c "$CONF" restart tr-agent
    ;;
  status)
    $CTL -c "$CONF" status tr-agent
    ;;
  logs)
    $CTL -c "$CONF" tail -f tr-agent
    ;;
  shutdown)
    $CTL -c "$CONF" shutdown
    echo "supervisord stopped."
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|shutdown}"
    exit 1
    ;;
esac
