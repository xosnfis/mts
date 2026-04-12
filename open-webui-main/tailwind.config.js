import typography from '@tailwindcss/typography';
import containerQueries from '@tailwindcss/container-queries';

/** @type {import('tailwindcss').Config} */
export default {
	darkMode: 'class',
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {
			colors: {
				'mts-red': '#E30613',
				'mts-red-dark': '#C0050F',
				'mts-gray-bg': '#F2F3F7',
				'mts-dark': '#1D2023'
			},
			borderRadius: {
				'2xl': '1rem',
				'3xl': '1.25rem',
				'4xl': '1.5rem'
			},
			typography: {
				DEFAULT: {
					css: {
						pre: false,
						code: false,
						'pre code': false,
						'code::before': false,
						'code::after': false
					}
				}
			},
			padding: {
				'safe-bottom': 'env(safe-area-inset-bottom)'
			},
			transitionProperty: {
				width: 'width'
			}
		}
	},
	plugins: [typography, containerQueries]
};
