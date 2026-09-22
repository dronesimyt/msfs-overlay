# MSFS Overlay

A live flight information overlay for Microsoft Flight Simulator 2024! Perfect for streamers who want to display real-time flight data, flight plans, and beautiful airline branding in their streams.

## What Does It Do?

✈️ **Shows Your Flight:**

- Altitude, speed, heading, and climb rate
- Wind and temperature information
- Where you are on your flight plan with progress tracking

📋 **Loads Your Flight Plan:**

- Automatically grabs your flight plan from SimBrief
- Shows origin, destination, and flight route
- Displays airport information (METAR, planned runway)

🎨 **Airline-Branded Look:**

- Automatically matches the theme to your airline
- 10+ airline themes (Delta, Lufthansa, United, etc.)
- Beautiful colors and logos

📱 **Social Media Integration:**

- Displays your TikTok follower count
- Perfect for content creators and streamers

## Quick Start

### What You Need

- Windows 10/11
- Microsoft Flight Simulator 2024
- Python 3.13 (or later)
- A SimBrief account (free at simbrief.com)

### Step 1: Download and Set Up

Download or clone this project, then install the required packages:

```bash
pip install -r requirements.txt
```

### Step 2: Settings

Copy `config.example.json` to `config.json` and fill in your information:

```json
{
  "simbrief_username": "your_simbrief_username",
  "simbrief_userid": "12345"
}
```

📝 **How to find your SimBrief info:**

1. Go to simbrief.com and log in
2. Your username is what you use to log in
3. Your user ID is in your profile page URL (after `/pilot/`)

### Step 3: Start It Up

Double-click `start_overlay.bat` and wait for it to start.

### Step 4: Add to Your Stream

In OBS or Streamlabs:

1. Add a **Browser source**
2. Enter this URL: `http://localhost:5000`
3. Set the width to about 1200px
4. Done!

Now when you fly in MSFS, the overlay will show all your flight data. Load a flight plan in SimBrief for maximum information!

## The Display Shows

- **Flight Info**: altitude, speed, heading, temperature
- **Navigation**: how far you are along your flight plan
- **Airports**: origin and destination details
- **Your Airline Theme**: automatically changes based on your flight plan
- **TikTok Followers**: (if configured)

## Airlines Included

- Delta Air Lines (DAL)
- Lufthansa (DLH)
- United Airlines (UAL)
- EasyJet (EZY)
- Iberia (IBE)
- Eurowings (EWG)
- Royal Jordanian (RJA)
- Vueling (VKG)
  ... and more!

## Optional: Social Media Setup

Want to show your TikTok followers? Add these to `config.json`:

```json
{
  "tiktok_username": "your_tiktok_name_without_at",
  "tiktok_user_id": "your_tiktok_user_id",
  "tiktok_followers_goal": 500
}
```

`tiktok_username` is your @name without the @, `tiktok_user_id` is the number you find on your profile page URL (only needed for the tokcount fallback).

The count is read from your public TikTok profile page, with tokcount.com as a fallback. Both are unofficial and may stop working at any time, so the last known value is cached and shown dimmed once it gets old. `"followers_enabled": false` turns the whole thing off, `"followers_refresh_seconds"` sets how often it is refreshed (default 60).

## Having Issues?

**The overlay isn't showing in OBS:**

- Make sure the application is running (it runs hidden; `start_overlay.bat` starts it, `stop_overlay.bat` stops it)
- Try paste the URL `http://localhost:5000` in your browser first to test
- Restart OBS and refresh the browser source

**Flight data not showing:**

- Make sure MSFS 2024 is running
- Load a flight plan in SimBrief first
- Try clicking refresh in your browser source

**SimBrief data not loading:**

- Double-check your username and user ID in `config.json`
- Make sure you have a flight plan loaded in SimBrief
- Sometimes SimBrief takes a few seconds to respond

**TikTok followers not showing:**

- Verify your TikTok user ID is correct
- Make sure the internet connection is working

## Need Help?

Check your `config.json` against `config.example.json`. Every setting can also be provided as an upper-case environment variable (e.g. `SIMBRIEF_USERID`). Logs are written to `overlay.log`.
