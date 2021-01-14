import { Component, OnInit } from '@angular/core';
import { ApiService } from '../api.service';
import { ApiResult } from '../api-result';

@Component({
  selector: 'app-identify',
  templateUrl: './identify.component.html',
  styleUrls: ['./identify.component.scss']
})
export class IdentifyComponent implements OnInit {

  constructor(private apiService: ApiService) { }

  ERR_MISSING_IMAGE = "Aucune image n'a été chargée"

  uploadedFileName: String = ''
  uploadedImg: String = ''
  resultImg: String = ''

  errMessage: String = ''

  ngOnInit(): void {
  }

  onSubmit() {
    if (!this.uploadedFileName) {
      this.errMessage = this.ERR_MISSING_IMAGE
    }

    this.apiService.identify({ 'img': this.uploadedImg })
      .subscribe(
        (result: ApiResult) => this.resultImg = "data:image/jpeg;base64," + result.img,
        (httpError) => this.errMessage = httpError.error.errorMsg
      );

    this.uploadedImg = ''
    this.uploadedFileName = ''
  }

  onFileSelected(event: any) {
    this.errMessage = ''
    this.resultImg = ''

    var file = event.target.files[0];

    var reader = new FileReader();
    reader.onloadend = () => {
      console.log('RESULT', reader.result)
      this.uploadedImg = String(reader.result)
    }

    reader.readAsDataURL(file);
  }

}
