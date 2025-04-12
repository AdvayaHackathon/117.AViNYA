<?php
/**
 * The base configuration for WordPress
 *
 * The wp-config.php creation script uses this file during the installation.
 * You don't have to use the website, you can copy this file to "wp-config.php"
 * and fill in the values.
 *
 * This file contains the following configurations:
 *
 * * Database settings
 * * Secret keys
 * * Database table prefix
 * * ABSPATH
 *
 * @link https://developer.wordpress.org/advanced-administration/wordpress/wp-config/
 *
 * @package WordPress
 */

// ** Database settings - You can get this info from your web host ** //
/** The name of the database for WordPress */
define( 'DB_NAME', 'avinya' );

/** Database username */
define( 'DB_USER', 'root' );

/** Database password */
define( 'DB_PASSWORD', '' );

/** Database hostname */
define( 'DB_HOST', 'localhost' );

/** Database charset to use in creating database tables. */
define( 'DB_CHARSET', 'utf8mb4' );

/** The database collate type. Don't change this if in doubt. */
define( 'DB_COLLATE', '' );

/**#@+
 * Authentication unique keys and salts.
 *
 * Change these to different unique phrases! You can generate these using
 * the {@link https://api.wordpress.org/secret-key/1.1/salt/ WordPress.org secret-key service}.
 *
 * You can change these at any point in time to invalidate all existing cookies.
 * This will force all users to have to log in again.
 *
 * @since 2.6.0
 */
define( 'AUTH_KEY',         ')A+FUNuqiV/sy^M/5Y[H|<W@` ^2(vTauRn>)[`G$0Ab&(DQ2hHu]RrSxJ;m-k;z' );
define( 'SECURE_AUTH_KEY',  'Fe%e+;S|RgS*2X1$e2J>@/X~-,m~qkFxdLFe8av%BnxY6W-|$#q~/V_la.MrZ;ET' );
define( 'LOGGED_IN_KEY',    'zvG*O9`Mjfqb)?-#JgmoT[a@E_dS^i$PVb3k1jFfq_S[Ql3^xs2#*J0Zc&#.Px+&' );
define( 'NONCE_KEY',        'gcDKo<ev]xC&i,CY4)MdJGiQ_l_EyI1[{zJeX}NBScr4_3As?#a,^ny][{vJ?=]r' );
define( 'AUTH_SALT',        'n=;<3Ma}?|lnZ=C*cZk[Sw0Z_BuB4GQkU^[dr^)JL&hJrC%]u(B2X}9QpDFbz/CF' );
define( 'SECURE_AUTH_SALT', '@<b9~%|f&MfDMxVxLg!PMhtvs^tRd94H 4uK!GXz|~Rlzoj!G5I%6*}lpu-DiN>V' );
define( 'LOGGED_IN_SALT',   '0@!&h JZ/XwCq@|I;@Z_@gHsx1H7=s[GD~7.jR9lSZhKMD?Dw;mTMlchF!AZ7U$e' );
define( 'NONCE_SALT',       'HoGQ+UXT_)R/bpD$:Uw}ra)3<CTJv(Hx0Csao_X,+[t8q0Wx+64(tbSQ4wq82twn' );

/**#@-*/

/**
 * WordPress database table prefix.
 *
 * You can have multiple installations in one database if you give each
 * a unique prefix. Only numbers, letters, and underscores please!
 *
 * At the installation time, database tables are created with the specified prefix.
 * Changing this value after WordPress is installed will make your site think
 * it has not been installed.
 *
 * @link https://developer.wordpress.org/advanced-administration/wordpress/wp-config/#table-prefix
 */
$table_prefix = 'wp_';

/**
 * For developers: WordPress debugging mode.
 *
 * Change this to true to enable the display of notices during development.
 * It is strongly recommended that plugin and theme developers use WP_DEBUG
 * in their development environments.
 *
 * For information on other constants that can be used for debugging,
 * visit the documentation.
 *
 * @link https://developer.wordpress.org/advanced-administration/debug/debug-wordpress/
 */
define( 'WP_DEBUG', false );

/* Add any custom values between this line and the "stop editing" line. */



/* That's all, stop editing! Happy publishing. */

/** Absolute path to the WordPress directory. */
if ( ! defined( 'ABSPATH' ) ) {
	define( 'ABSPATH', __DIR__ . '/' );
}

/** Sets up WordPress vars and included files. */
require_once ABSPATH . 'wp-settings.php';
