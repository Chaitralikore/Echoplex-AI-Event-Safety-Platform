/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        base: '#121212',
        surface: '#181818',
        surfaceHover: '#282828',
        accent: '#1DB954',
        textPrimary: '#FFFFFF',
        textSecondary: '#B3B3B3',
      },
    },
  },
  plugins: [],
};
