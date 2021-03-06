const http = require('http');
/**
 * get - wrapper over http requester
 *
 * @param  {String} url description
 * @return {Promise}     description
 */
function get(url) {
  return new Promise(function(resolve, reject) {
    http.get(url, (res) => {
      const {statusCode} = res;

      let error;
      if (statusCode !== 200) {
        error = new Error('Request Failed.\n' +
                     `Status Code: ${statusCode}`);
      }
      if (error) {
        console.error(error.message);
        // consume response data to free up memory
        res.resume();
        reject(error);
      }

      res.setEncoding('utf8');
      let rawData = '';
      res.on('data', (chunk) => {
        rawData += chunk;
      });
      res.on('end', () => {
        try {
          const parsedData = JSON.parse(rawData);
          resolve(parsedData);
        } catch (e) {
          console.error(e.message);
        }
      });
    }).on('error', (e) => {
      console.error(`Got error: ${e.message}`);
      reject(e);
    });
  });
}

module.exports = {
  get
};
