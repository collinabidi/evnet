# do whatever we need to do to start this thingy

mkdir staging

(killall -v node || true) && node ./index.js
