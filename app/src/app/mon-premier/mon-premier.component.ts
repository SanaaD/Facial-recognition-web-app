import { Component, OnInit } from '@angular/core';
import { ApiService } from '../api.service';

@Component({
  selector: 'app-mon-premier',
  templateUrl: './mon-premier.component.html',
  styleUrls: ['./mon-premier.component.scss']
})
export class MonPremierComponent implements OnInit {

  constructor(private apiService: ApiService) { }

  ERR_MISSING_NAME = "Aucun prénom n'a été saisi"
  ERR_MISSING_IMAGE = "Aucune image n'a été chargée"

  name: String = ''
  uploadedFileName: String = ''
  uploadedImg: String = ''
  errMessage: String = ''
  label: String = ''
  success = false


  ngOnInit(): void {
  }

  onSubmit() {

    this.errMessage = ''

    if (!this.name) {
      this.errMessage = this.ERR_MISSING_NAME
    }
    else if (!this.uploadedImg) {
      this.errMessage = this.ERR_MISSING_IMAGE
    }
    else {
      this.apiService.add({ 'name': this.name, 'img': this.uploadedImg })
        .subscribe(
          (result) => {
            this.success = true;
            this.label = this.name;
            this.name = ''
            console.log(this.label)
            setTimeout(() => { this.success = false; this.label = '' }, 3000);
          },
          (httpError) => this.errMessage = httpError.error.errorMsg
        );
      this.uploadedImg = ''
      this.uploadedFileName = ''
    }
  }

  onFileSelected(event: any) {
    if (this.errMessage == this.ERR_MISSING_IMAGE)
      this.errMessage = ''

    var file = event.target.files[0];

    var reader = new FileReader();
    reader.onloadend = () => {
      this.uploadedImg = String(reader.result)
    }

    reader.readAsDataURL(file);
  }
}
