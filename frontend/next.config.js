const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fix: multiple lockfiles warning — explicitly tell Next.js this frontend
  // folder is the root, not the parent GEN_AI_PY folder
  outputFileTracingRoot: path.join(__dirname),

  // Silence the workspace root detection warning
  experimental: {},

  async rewrites() {
    return [
      // Proxy API calls through Next.js to avoid CORS issues
      // Uncomment if you hit CORS errors:
      // {
      //   source: '/api/:path*',
      //   destination: 'http://localhost:8000/api/:path*',
      // },
    ];
  },
};

module.exports = nextConfig;
