/** @type {import('next').NextConfig} */
const apiUrl = process.env.API_URL || 'http://127.0.0.1:8000';

const nextConfig = {
  experimental: {
    // Classical DIP on high-res (e.g. 4288x2848) fundus images can exceed Next's 30s default proxyTimeout
    proxyTimeout: 300_000,
  },
  async rewrites() {
    return [
      {
        source: '/api/py/:path*',
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
