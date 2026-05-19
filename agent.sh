#!/usr/bin/env bash
# Control the tr-agent supervisord processes.
# Usage: ./agent.sh {start|stop|restart|status|logs|web-logs|shutdown}

CONF="$(dirname "$0")/supervisord.conf"
CTL=".venv/bin/supervisorctl"
SD=".venv/bin/supervisord"

cd "$(dirname "$0")" || exit 1

case "${1:-status}" in
  start)
    if $CTL -c "$CONF" status tr-agent &>/dev/null 2>&1; then
      echo "Already running. Use './agent.sh restart' to reload."
    else
      $SD -c "$CONF"
      sleep 1
      $CTL -c "$CONF" status
      echo ""
      echo "Dashboard → http://0.0.0.0:8080"
    fi
    ;;
  stop)
    $CTL -c "$CONF" stop tr-agent tr-agent-web
    ;;
  restart)
    $CTL -c "$CONF" restart tr-agent tr-agent-web
    ;;
  status)
    $CTL -c "$CONF" status
    ;;
  logs)
    $CTL -c "$CONF" tail -f tr-agent
    ;;
  web-logs)
    $CTL -c "$CONF" tail -f tr-agent-web
    ;;
  shutdown)
    $CTL -c "$CONF" shutdown
    echo "supervisord stopped."
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|web-logs|shutdown}"
    exit 1
    ;;
esac
