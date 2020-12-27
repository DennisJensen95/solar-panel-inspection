import React, { Component, createRef } from 'react';
import axios from 'axios';
import './App.css'

class App extends Component {

  state = {
    title: '',
    content: '',
    image: null,
    file: null,
    result: null,
    imgData: createRef(),
  };

  handleImageChange = (e) => {
    const img = e.target.files[0];
    console.log(this.state.imgData)
    console.log(img);
    this.setState({
      image: img,
      file: URL.createObjectURL(img),
    })
  };

  handleSubmit = (e) => {
    e.preventDefault();
    console.log(this.state);
    let form_data = new FormData();
    form_data.append('image', this.state.image, this.state.image.name);
    let url = 'http://localhost:8000/image/';
    axios.post(url, form_data)
        .then(res => {
          this.setState({result:  res.data.encoded_image});
        })
        .catch(err => console.log(err))
  };

  render() {
    return (
      <div className="App">
        <div class="header-div">
          <text class="header">
            Solar panel inspection
          </text>
        </div>
        <div class="header-div-2">
          <text class="header-2">
            Import and image of a solar panel to get it analyzed
          </text>
        </div>
        <form onSubmit={this.handleSubmit}>
          <p>
            <input type="file"
                   class="image"
                   accept="image/png, image/jpeg"  onChange={this.handleImageChange} required/>
            <img src={this.state.file} width="150" height="100"/>
          </p>
          <input type="submit"/>
        </form>
        <img class="image result" src={this.state.result} width="600" height="400"/>
      </div>
    );
  }
}

export default App;