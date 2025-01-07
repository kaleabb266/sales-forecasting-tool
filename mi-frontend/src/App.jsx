import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AppBar, Toolbar, Typography, Container, Box, MenuItem, Select, InputLabel, FormControl, Button, Drawer, List, ListItem, ListItemText } from '@mui/material';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [store, setStore] = useState('');
  const [stores, setStores] = useState([]);  // Initialize as an empty array for fetched stores
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // Fetch stores from the backend API when the component loads
  useEffect(() => {
    const fetchStores = async () => {
      try {
        const response = await axios.get('http://localhost:8000/stores/');
        setStores(response.data.stores);  // Set the fetched stores into state
      } catch (err) {
        setError("Error fetching stores");
      }
    };

    fetchStores();
  }, []);  // Empty dependency array ensures this runs only once when the component mounts

  useEffect(() => {
    if (store) {
      fetchPrediction(store);
    }
  }, [store]);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const fetchPrediction = async (store) => {
    try {
      const inputData = [658, -4364, 6102, -1091, -229, 238, 243]; // Example data, adjust based on real input

      const response = await axios.post('http://localhost:8000/predict/', {
        features: inputData
      });

      setPrediction(response.data.predictions);
    } catch (err) {
      setError("Error making prediction");
    }
  };

  const handleStoreChange = (event) => {
    setStore(event.target.value);
  };

  return (
    <div className="App">
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Rossmann Pharmaceuticals Sales Prediction
          </Typography>
          <Button color="inherit" onClick={toggleSidebar}>Toggle Sidebar</Button>
        </Toolbar>
      </AppBar>

      {/* Sidebar */}
      <Drawer anchor="left" open={isSidebarOpen} onClose={toggleSidebar}>
        <List>
          {stores.map((store, index) => (
            <ListItem button key={store} onClick={() => setStore(store)}>
              <ListItemText primary={`Store ${store}`} />
            </ListItem>
          ))}
        </List>
      </Drawer>

      <Container>
        <Box mt={4} mb={2}>
          <Typography variant="h4">Business Need</Typography>
          <Typography variant="body1" paragraph>
          Rossmann Pharmaceuticals' innovative web interface offers precise six-week sales forecasts for all stores, empowering our managers to make data-driven decisions and optimize operations. By leveraging cutting-edge machine learning technology, this tool provides accurate predictions based on historical sales data, store-specific factors, and market trends. With intuitive visualizations and real-time updates, our managers can confidently plan inventory, allocate resources, and ensure customer satisfaction.
          </Typography>
        </Box>

        {/* Store Selection */}
        <Box mt={4} mb={4}>
          <FormControl fullWidth>
            <InputLabel id="store-select-label">Select Store</InputLabel>
            <Select
              labelId="store-select-label"
              id="store-select"
              value={store}
              label="Store"
              onChange={handleStoreChange}
            >
              {stores.map((store) => (
                <MenuItem key={store} value={store}>
                  Store {store}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        {/* Prediction Result */}
        {prediction && (
          <Box mt={4}>
            <Typography variant="h6">Prediction for Store {store}:</Typography>
            <div>
              <h2>Sales Predictions (Next 6 Weeks):</h2>
              {prediction.map((p, index) => (
                <Typography key={index} variant="body1">Day {index + 1}: {p}</Typography>
              ))}
            </div>
          </Box>
        )}

        {error && <Typography color="error">{error}</Typography>}
      </Container>
    </div>
  );
}

export default App;
