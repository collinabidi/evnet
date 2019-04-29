const app = require('express')();
const {spawn} = require('child_process');
const {get} = require('requester');
const fs = require('fs');
 const shell = require('shelljs');

async function init(){
   console.log(1)
   await sleep(1000)
   console.log(2)
}
function sleep(ms){
    return new Promise(resolve=>{
        setTimeout(resolve,ms)
    })
}

// run until done
let done = false;
let lock = false;

const TFLITE_NETWORK_PATH = '../src/dump';
const HPEC_USERNAME = 'pi';
const HPEC_DUMP_FILENAME = 'dump';

const HPC_STAGING_PATH = './staging';
const HPEC_STAGING_PATH = `/home/${HPEC_USERNAME}/staging/${HPEC_DUMP_FILENAME}`;

const HPC_2_HPEC_PORT = 3000;
const HPEC_TO_HPC_PORT = 4000;

const HPEC_IP = `192.168.1.1`; // NOTE: NEED TO CHANGE THIS
const HPC_IP = `192.168.1.1`; // NOTE: NEED TO CHANGE THIS

function scp_response((req, res)) {
  // transfer file back from hpec
  const scp = execFileSync('scp', [`${HPEC_USERNAME}@${HPEC_IP}:${HPEC_STAGING_PATH}`], [HPEC_DUMP_FILENAME]);
  lock = false;
  // ANY LOGIC CHECKING IF WE ARE DONE GOES HERE
  res.sendStatus(200);
}

app.route('/').get(scp_response);

app.listen(HPC_2_HPEC_PORT, (err) => {
  console.log('PI IS LISTENING ON SERVER ON PORT', PORT);

  while (!done) {
    // create child process for evnet
    const evnet = execFileSync('python', ['../src/main.py']);
    // net is trained, move file to staging area
    const mov = execFileSync('mv', [TFLITE_NETWORK_PATH, HPC_STAGING_PATH]);
    // PING!
    const ret = await get(HPEC_IP + ':' + HPC_2_HPEC_PORT);

    lock = true;

    // wait until we get a return from the hpec system
    while(lock) {sleep(10)};
    // check if done!
  }

});
