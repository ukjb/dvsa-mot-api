# DVSA MOT API Service

This service provides MOT data and vehicle rarity scores using the DVSA API.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dvsa-mot-api.git
cd dvsa-mot-api
```

2. Create and configure your environment file:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

3. Run the installation script:
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

## Deployment

The service will automatically start on boot. You can manage it using:
```bash
sudo systemctl start|stop|restart|status dvsa-api
```

## Updating

To update the service:
1. Pull the latest changes:
```bash
git pull origin main
```

2. Restart the service:
```bash
sudo systemctl restart dvsa-api
```

## Logs

View service logs:
```bash
sudo journalctl -u dvsa-api
```
