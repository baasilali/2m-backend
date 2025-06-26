const https = require('https');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

// Configuration
const API_HOST = 'api.skinport.com';
const API_PATH = '/v1/items'; // default parameters can be appended if needed
const OUTPUT_FILE = path.join(__dirname, 'data', 'skinport_data.json');

/**
 * Fetch data from Skinport API with Brotli compression.
 * Returns a Promise that resolves to the parsed JSON response.
 */
function fetchSkinportItems() {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: API_HOST,
      path: API_PATH,
      method: 'GET',
      headers: {
        'Accept-Encoding': 'br'
      }
    };

    const req = https.request(options, res => {
      const chunks = [];

      res.on('data', chunk => chunks.push(chunk));

      res.on('end', () => {
        try {
          const compressedBuffer = Buffer.concat(chunks);
          // Attempt Brotli decompression. If it fails, assume data is already decompressed.
          let decompressedBuffer;
          try {
            decompressedBuffer = zlib.brotliDecompressSync(compressedBuffer);
          } catch (err) {
            decompressedBuffer = compressedBuffer;
          }

          const result = JSON.parse(decompressedBuffer.toString());
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
    });

    req.on('error', reject);
    req.end();
  });
}

/**
 * Write data to the output JSON file.
 */
function writeToFile(data) {
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(data, null, 2));
  console.log(`Skinport data updated → ${OUTPUT_FILE}`);
}

(async () => {
  try {
    console.log('Fetching latest Skinport items...');
    const items = await fetchSkinportItems();
    writeToFile(items);
  } catch (err) {
    console.error('Failed to update Skinport data:', err);
    process.exit(1);
  }
})(); 